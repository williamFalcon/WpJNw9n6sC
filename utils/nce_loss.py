import torch
from torch import nn


class CPCLossNCE(nn.Module):

    def nce_loss(self, anchor, pos_scores, negative_samples, mask_mat):

        # RKHS = embedding dim
        pos_scores = pos_scores.float()
        batch_size, emb_dim = anchor.size()
        nb_feat_vectors = negative_samples.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(anchor, negative_samples)
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors).float()

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # (batch_size, batch_size, nb_feat_vectors)
        neg_scores = (mask_neg * raw_scores)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores

    def forward(self, Z, C, W_list):
        """

        :param Z: latent vars (b*patches, emb_dim, h, w)
        :param C: context var (b*patches, emb_dim, h, w)
        :param W_list: list of k-1 W projections
        :return:
        """
        # (b, dim, w. h)
        batch_size, emb_dim, h, w = Z.size()

        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(Z.device.index)
        diag_mat = diag_mat.float()

        losses = []
        # calculate loss for each k

        Z_neg = Z.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        for i in range(0, h-1):
            for j in range(0, w):
                cij = C[:, :, i, j]

                for k in range(i+1, h):
                    Wk = W_list[str(k)]

                    z_hat_ik_j = Wk(cij)

                    zikj = Z[:, :, k, j]

                    # BATCH DOT PRODUCT
                    # (b, d) x (b, d) -> (b, 1)
                    pos_scores = torch.bmm(z_hat_ik_j.unsqueeze(1), zikj.unsqueeze(2))
                    pos_scores = pos_scores.squeeze(-1).squeeze(-1)

                    loss = self.nce_loss(z_hat_ik_j, pos_scores, Z_neg, diag_mat)
                    losses.append(loss)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss
