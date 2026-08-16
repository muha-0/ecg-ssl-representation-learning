import torch

def collate_patient_pairs(batch):
    xs = []
    pair_ids = []
    patient_ids = []
    rec1_names = []
    rec2_names = []
    ys = []

    for i, item in enumerate(batch):
        xs.append(item["x1"])
        xs.append(item["x2"])
        pair_ids.extend([i, i])
        patient_ids.append(item["patient_id"])
        rec1_names.append(item.get("rec1", item.get("rec", "NA")))
        rec2_names.append(item.get("rec2", item.get("rec", "NA")))
        ys.append(item["y"])

    x = torch.stack(xs, dim=0)
    pair_ids = torch.tensor(pair_ids, dtype=torch.long)
    return {
        "x": x,
        "pair_ids": pair_ids,
        "patient_ids": patient_ids,
        "rec1": rec1_names,
        "rec2": rec2_names,
        "y": ys,
    }

def collate_tenmin(batch):
    # batch: list of dicts, each x16 is [37, T16]
    x16 = torch.stack([b["x"] for b in batch], dim=0)  # [B, 37, T16]
    y10 = torch.stack([b["y"] for b in batch], dim=0)  # [B]
    return {"x": x16, "y": y10}

def collate_patient_windows(batch):
    # flatten to windows: [sumK, T] and keep patient ids
    xs = []
    pids = []
    for item in batch:
        xk = item["x"]  # [K, T]
        K = xk.size(0)
        xs.append(xk)
        pids.extend([item["patient_id"]] * K)
    x = torch.cat(xs, dim=0)  # [B*K, T]
    return {"x": x, "patient_ids": pids}