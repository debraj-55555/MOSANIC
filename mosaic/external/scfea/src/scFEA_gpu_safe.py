#!/usr/bin/env python3
"""
GPU-safe scFEA runner — keeps X on CPU, uses mini-batch training.
Drop-in replacement for scFEA.py main() that avoids GPU OOM
when n_cells × n_overlap_genes × n_modules is large.

Key changes from original:
  1. X, geneExprScale, module_scale stay on CPU
  2. BATCH_SIZE = min(n_cells, 4096) instead of n_cells
  3. ClassFlux.updateC creates c on correct device
  4. DataLoader sends CPU batches; .to(device) in loop moves to GPU
"""

import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm

# scFEA imports (relative to scFEA/src/)
sys.path.insert(0, os.path.dirname(__file__))
from ClassFlux import FLUX
from util import pearsonr
from DatasetFlux import MyDataset

# ── hyper-parameters (same as original) ──────────────────────────
LEARN_RATE = 0.008
LAMB_BA = 1
LAMB_NG = 1
LAMB_CELL = 1
LAMB_MOD = 1e-2
MAX_BATCH = 4096          # <-- new: cap per-batch size


def myLoss(m, c, lamb1=0.2, lamb2=0.2, lamb3=0.2, lamb4=0.2,
           geneScale=None, moduleScale=None):
    """Same as original but creates tensors on the correct device."""
    device = m.device

    total1 = torch.sum(torch.pow(c, 2), dim=1)
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)

    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    if (diff > 0).sum() == m.shape[0]:
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff

    if lamb4 > 0:
        corr = torch.ones(m.shape[0], device=device)
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        total4 = torch.ones(m.shape[0], device=device) - corr
    else:
        total4 = torch.zeros(m.shape[0], device=device)

    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)
    return loss1 + loss2 + loss3 + loss4, loss1, loss2, loss3, loss4


class FLUX_SAFE(FLUX):
    """Patched FLUX that creates intermediate tensors on the correct device."""

    def updateC(self, m, n_comps, cmMat):
        c = torch.zeros((m.shape[0], n_comps), device=m.device)
        for i in range(c.shape[1]):
            tmp = m * cmMat[i, :]
            c[:, i] = torch.sum(tmp, dim=1)
        return c


def main(args):
    data_path  = args.data_dir
    input_path = args.input_dir
    res_dir    = args.res_dir
    os.makedirs(res_dir, exist_ok=True)
    test_file       = args.test_file
    moduleGene_file = args.moduleGene_file
    cm_file         = args.stoichiometry_matrix
    sc_imputation   = args.sc_imputation
    cName_file      = args.cName_file
    fileName        = args.output_flux_file
    balanceName     = args.output_balance_file
    EPOCH           = args.train_epoch

    if EPOCH <= 0:
        raise ValueError("EPOCH must be > 0")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & impute ────────────────────────────────────────────
    print("Starting load data...")
    geneExpr = pd.read_csv(f"{input_path}/{test_file}", index_col=0).T * 1.0

    if sc_imputation:
        import magic
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpr = magic_operator.fit_transform(geneExpr)

    if geneExpr.max().max() > 50:
        geneExpr = (geneExpr + 1).apply(np.log2)

    geneExprSum   = geneExpr.sum(axis=1)
    stand         = geneExprSum.mean()
    geneExprScale = torch.FloatTensor((geneExprSum / stand).values)   # CPU!

    moduleGene = pd.read_csv(f"{data_path}/{moduleGene_file}", sep=",", index_col=0)
    moduleLen  = np.array([moduleGene.iloc[i, :].notna().sum()
                           for i in range(moduleGene.shape[0])])

    # gene overlap
    module_gene_all = set()
    for i in range(moduleGene.shape[0]):
        for j in range(moduleGene.shape[1]):
            v = moduleGene.iloc[i, j]
            if pd.notna(v):
                module_gene_all.add(v)
    gene_overlap = sorted(set(geneExpr.columns) & module_gene_all)

    cmMat = pd.read_csv(f"{data_path}/{cm_file}", sep=",", header=None).values
    cmMat = torch.FloatTensor(cmMat).to(device)            # small, keep on GPU

    cName = None
    if cName_file != "noCompoundName":
        print("Load compound name file.")
        cName = pd.read_csv(f"{data_path}/{cName_file}", sep=",", header=0).columns
    print("Load data done.")

    # ── Process: build X ─────────────────────────────────────────
    print("Starting process data...")
    emptyNode = []
    geneExpr = geneExpr[gene_overlap]
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    n_modules = moduleGene.shape[0]
    n_genes   = len(gene_names)
    n_cells   = len(cell_names)
    n_comps   = cmMat.shape[0]

    # Estimate memory to decide batch approach
    est_x_gb = n_cells * n_modules * n_genes * 4 / 1e9
    print(f"  n_cells={n_cells}, n_genes={n_genes}, n_modules={n_modules}")
    print(f"  Estimated X size: {est_x_gb:.2f} GiB (float32)")

    geneExprDf = pd.DataFrame(columns=["Module_Gene"] + list(cell_names))
    for i in range(n_modules):
        genes = [g for g in moduleGene.iloc[i, :].values.astype(str) if g != "nan"]
        if not genes:
            emptyNode.append(i)
            continue
        temp = geneExpr.copy()
        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
        temp = temp.T
        temp["Module_Gene"] = ["%02d_%s" % (i, g) for g in gene_names]
        geneExprDf = geneExprDf._append(temp, ignore_index=True, sort=False)

    geneExprDf.index = geneExprDf["Module_Gene"]
    geneExprDf.drop("Module_Gene", axis="columns", inplace=True)

    # X stays on CPU
    X = torch.FloatTensor(geneExprDf.values.astype("float32").T)    # CPU!
    print(f"  X shape: {X.shape}, device: {X.device}")

    df = geneExprDf.copy()
    df.index = [i.split("_")[0] for i in df.index]
    df.index = df.index.astype(int)
    module_scale = df.groupby(df.index).sum().T
    module_scale = torch.FloatTensor(module_scale.values / moduleLen)  # CPU!

    # Free the big DataFrame
    del geneExprDf, df
    print("Process data done.")

    # ── NN training (mini-batch) ─────────────────────────────────
    BATCH_SIZE = min(n_cells, MAX_BATCH)
    print(f"Training: EPOCH={EPOCH}, BATCH_SIZE={BATCH_SIZE}, "
          f"LR={LEARN_RATE}")

    torch.manual_seed(16)
    net = FLUX_SAFE(X, n_modules, f_in=n_genes, f_out=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE)

    dataloader_params = {"batch_size": BATCH_SIZE, "shuffle": False,
                         "num_workers": 0, "pin_memory": True}
    train_set    = MyDataset(X, geneExprScale, module_scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               **dataloader_params)

    print("Starting train neural network...")
    start = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    lossName = f"{res_dir}/lossValue_{timestr}.txt"
    file_loss = open(lossName, "a")

    net.train()
    for epoch in tqdm(range(EPOCH), desc="scFEA"):
        loss, loss1, loss2, loss3, loss4 = 0, 0, 0, 0, 0
        for X_b, Xsc_b, msc_b in train_loader:
            X_batch   = X_b.float().to(device)
            Xsc_batch = Xsc_b.float().to(device)
            msc_batch = msc_b.float().to(device)

            out_m, out_c = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            lb, l1b, l2b, l3b, l4b = myLoss(
                out_m, out_c,
                lamb1=LAMB_BA, lamb2=LAMB_NG,
                lamb3=LAMB_CELL, lamb4=LAMB_MOD,
                geneScale=Xsc_batch, moduleScale=msc_batch)

            optimizer.zero_grad()
            lb.backward()
            optimizer.step()

            loss  += lb.item()
            loss1 += l1b.item()
            loss2 += l2b.item()
            loss3 += l3b.item()
            loss4 += l4b.item()

        file_loss.write(
            f"epoch: {epoch+1:02d}, loss1: {loss1:.8f}, loss2: {loss2:.8f}, "
            f"loss3: {loss3:.8f}, loss4: {loss4:.8f}, loss: {loss:.8f}.\n")

    end = time.time()
    print(f"Training time: {end - start:.1f}s")
    file_loss.close()

    # ── Inference (batch_size=1 to save GPU memory) ──────────────
    test_params = {"batch_size": min(BATCH_SIZE, 1024), "shuffle": False,
                   "num_workers": 0, "pin_memory": True}
    test_set    = MyDataset(X, geneExprScale, module_scale)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, **test_params)

    fluxStatuTest = np.zeros((n_cells, n_modules), dtype="f")
    balanceStatus = np.zeros((n_cells, n_comps), dtype="f")
    net.eval()
    idx = 0
    with torch.no_grad():
        for X_b, Xsc_b, _ in test_loader:
            X_batch = X_b.float().to(device)
            out_m, out_c = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            bs = out_m.shape[0]
            fluxStatuTest[idx:idx+bs, :] = out_m.cpu().numpy()
            balanceStatus[idx:idx+bs, :] = out_c.cpu().numpy()
            idx += bs

    # ── Save ─────────────────────────────────────────────────────
    if fileName == "NULL":
        fileName = f"{res_dir}/{test_file[:-4]}.csv"
    setF = pd.DataFrame(fluxStatuTest, columns=moduleGene.index,
                         index=geneExpr.index.tolist())
    setF.to_csv(fileName)

    setB = pd.DataFrame(balanceStatus, index=setF.index)
    if cName is not None:
        setB.columns = cName
    if balanceName == "NULL":
        balanceName = f"{res_dir}/balance.csv"
    setB.to_csv(balanceName)

    print(f"scFEA job finished. Results in {res_dir}")


def parse_arguments(parser):
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--input_dir", type=str, default="input")
    parser.add_argument("--res_dir", type=str, default="output")
    parser.add_argument("--test_file", type=str, default="Melissa_full.csv")
    parser.add_argument("--moduleGene_file", type=str,
                        default="module_gene_m168.csv")
    parser.add_argument("--stoichiometry_matrix", type=str,
                        default="cmMat_complete_mouse_c70_m168.csv")
    parser.add_argument("--sc_imputation", type=str, default="True")
    parser.add_argument("--cName_file", type=str,
                        default="noCompoundName")
    parser.add_argument("--output_flux_file", type=str, default="NULL")
    parser.add_argument("--output_balance_file", type=str, default="NULL")
    parser.add_argument("--train_epoch", type=int, default=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_arguments(parser)
    args = parser.parse_args()

    # Fix sc_imputation to bool
    if isinstance(args.sc_imputation, str):
        args.sc_imputation = args.sc_imputation.lower() in ("true", "1", "yes")

    main(args)
