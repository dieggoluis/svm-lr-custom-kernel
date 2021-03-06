import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import regex
import substring_kernel as subker


def normalize_kernel(kernel):
    """
    Normalizes a kernel: kernel[x, y] by doing:
        kernel[x, y] / sqrt(kernel[x, x] * kernel[y, y])
    """

    nkernel = np.copy(kernel).astype(float)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    for i in range(nkernel.shape[0]):
        for j in range(i + 1, nkernel.shape[0]):
            q = np.sqrt(nkernel[i, i] * nkernel[j, j])
            if q > 0:
                nkernel[i, j] = float(nkernel[i, j])/q
                nkernel[j, i] = nkernel[i, j]  # symmetry

    # finally, set diagonal elements to 1
    np.fill_diagonal(nkernel, 1.)

    return nkernel


def spectrum_kernel(chainsA, chainsB):
    dna_dict = np.array(["".join(item) for item in itertools.product("ATCG", repeat=3)])
    chainsA = np.atleast_1d(chainsA)
    chainsB = np.atleast_1d(chainsB)
    matA = np.array([[chain[0].count(word) for chain in chainsA] for word in dna_dict])
    matB = np.array([[chain[0].count(word) for chain in chainsB] for word in dna_dict])
    return np.squeeze(np.dot(matA.T, matB))


def spectrum_norm_kernel(chainsA, chainsB):
    dna_dict = np.array(["".join(item) for item in itertools.product("ATCG", repeat=3)])
    chainsA = np.atleast_1d(chainsA)
    chainsB = np.atleast_1d(chainsB)
    matA = np.array([[chain[0].count(word) for chain in chainsA] for word in dna_dict])
    matB = np.array([[chain[0].count(word) for chain in chainsB] for word in dna_dict])
    return cosine_similarity(matA.T, matB.T)


def mismatch_kernel(chainsA, chainsB, k=3, m=0):
    chainsA = np.atleast_1d(chainsA)
    chainsB = np.atleast_1d(chainsB)

    NVocab = {}
    vocab = np.array(["".join(item) for item in itertools.product("ATCG", repeat=k)])
    n_vocab = len(vocab)
    idx = dict(zip(vocab, range(len(vocab))))
    all_seq = "".join(vocab)
    for kmer in vocab:
        neighbors = regex.findall("(" + kmer + ")" + "{s<=" + str(m) + "}", all_seq, overlapped=True)
        NVocab[kmer] = list(np.unique(neighbors))

    specA = []
    for chain in chainsA:
        spec = np.zeros(n_vocab)
        n = len(chain[0])
        for offset in range(n - k):
            kmer = chain[0][offset: offset + k]
            for nb in NVocab[kmer]:
                spec[idx[kmer]] += 1
        specA.append(spec)
    specA = np.asarray(specA)

    specB = []
    for chain in chainsB:
        spec = np.zeros(n_vocab)
        n = len(chain[0])
        for offset in range(n - k):
            kmer = chain[0][offset: offset + k]
            for nb in NVocab[kmer]:
                spec[idx[kmer]] += 1
        specB.append(spec)
    specB = np.asarray(specB)

    return cosine_similarity(specA, specB)


def substringKernel_cpp_nystrom(chainsA, chainsB, lbda=0.1, word_size=4,
                                sample_size=100):
    chainsA = np.atleast_1d(chainsA)
    sample = np.random.choice(chainsA.size, sample_size, replace=False)
    res = subker.getKernel(lbda, word_size, chainsA, chainsA[sample]).reshape(
        [chainsA.size, sample_size])
    mat = res[sample]
    w, v = np.linalg.eig(mat)
    w = w.real
    v = v.real
    ind = w > 1e-6
    v1 = v[:, ind]
    w1 = np.diag(1 / w[ind])
    almost_inverse = v1.dot(w1.dot(np.transpose(v1)))
    res = res.dot(almost_inverse.dot(np.transpose(res)))
    return np.squeeze(res)


def substringKernel_cpp(chainsA, chainsB ,lbda = 0.1, word_size = 4):
    chainsA = np.atleast_1d(chainsA)
    chainsB = np.atleast_1d(chainsB)
    res = subker.getKernel(lbda, word_size, chainsA, chainsB).reshape([
        chainsA.size, chainsB.size])
    return np.squeeze(res)
