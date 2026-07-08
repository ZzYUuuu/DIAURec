import torch as t
import torch.nn.functional as F
import numpy as np
import torch
import math


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss


def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()


def ssl_con_loss_all(x, y, z, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    z = F.normalize(z)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ z.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()


def ssl_con_loss_all2(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * x, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()


def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


def malignment(embedding_1, embedding_2, margin):
    embedding_1 = t.nn.functional.normalize(embedding_1, dim=-1)
    embedding_2 = t.nn.functional.normalize(embedding_2, dim=-1)

    cos_similarity = t.sum(embedding_1 * embedding_2, dim=-1)
    angle_ui = t.arccos(t.clamp(cos_similarity, -1 + 1e-7, 1 - 1e-7))

    angle_ui_plus_margin = angle_ui + (1 - t.sigmoid(margin))
    angle_ui_plus_margin = t.clamp(angle_ui_plus_margin, 0., np.pi)

    cos_similarity_margin = t.cos(angle_ui_plus_margin)
    return - cos_similarity_margin.mean()


def duniformity(embeddings):
    embeddings = t.nn.functional.normalize(embeddings, dim=-1)
    cos_similarity = t.nn.functional.cosine_similarity(embeddings[:, :, None], embeddings.t()[None, :, :])
    cos_similarity = t.tril(cos_similarity, diagonal=-1)
    cos_similarity = 2 - 2 * cos_similarity
    return cos_similarity.mul(-2).exp().mean().log()


# def coarse_match(x: t.Tensor, y: t.Tensor, eps: float = 1e-12) -> t.Tensor:
#     """
#     Coarse-grained Matching:
#     """
#     x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
#     return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def coarse_match(x, y, W, alpha=2, use_W=False, lambda_ortho=1.0, eps=1e-12):
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)

    if use_W:
        assert W is not None,
        y = F.normalize(y @ W.T, dim=-1, eps=eps)

    match = (x - y).norm(p=2, dim=1).pow(alpha).mean()

    if use_W:
        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        ortho = ((W.T @ W) - I).pow(2).sum()
        return match + lambda_ortho * ortho
    else:
        return match


def fine_match(x, y, candidate, alpha=2):
    embed1 = F.normalize(x, dim=1)
    embed2 = F.normalize(x, dim=1)
    similarity_matrix = torch.matmul(embed1, embed2.T)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    top_similar_indices = similarity_matrix.argmax(dim=1)
    y = y[top_similar_indices]

    x, y = x[candidate], y[candidate]
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    fine_mat = (x - y).norm(p=2, dim=1)
    return fine_mat.pow(alpha).mean()
