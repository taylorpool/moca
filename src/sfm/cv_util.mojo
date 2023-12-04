from utils.index import Index
import src.moca as mc
import math


# Stolen from: https://github.com/ftdlyc/PnP_Matlab/
fn rqGivens(
    A: Tensor[DType.float64],
    inout R: Tensor[DType.float64],
    inout Q: Tensor[DType.float64],
):
    var Ax = Tensor[DType.float64](3, 3)
    var Axy = Tensor[DType.float64](3, 3)
    var Axyz = Tensor[DType.float64](3, 3)
    var Gx = Tensor[DType.float64](3, 3)
    var Gy = Tensor[DType.float64](3, 3)
    var Gz = Tensor[DType.float64](3, 3)

    # 2nd - Set element 21 to zero
    if A[Index(2, 1)] == 0:
        Ax = A
        Gx = mc.eye(3)
    else:
        let r32 = math.sqrt(
            A[Index(2, 2)] * A[Index(2, 2)] + A[Index(2, 1)] * A[Index(2, 1)]
        )
        let c32 = A[Index(2, 2)] / r32
        let s32 = -A[Index(2, 1)] / r32
        Gx = mc.eye(3)
        Gx[Index(1, 1)] = c32
        Gx[Index(1, 2)] = -s32
        Gx[Index(2, 1)] = s32
        Gx[Index(2, 2)] = c32
        Ax = mc.mat_mat(A, Gx)

    # 2nd - Set element 20 to zero
    if A[Index(2, 0)] == 0:
        Axy = Ax
        Gy = mc.eye(3)
    else:
        let r31 = math.sqrt(
            Ax[Index(2, 2)] * Ax[Index(2, 2)] + Ax[Index(2, 0)] * Ax[Index(2, 0)]
        )
        let c31 = Ax[Index(2, 2)] / r31
        let s31 = Ax[Index(2, 0)] / r31
        Gy = mc.eye(3)
        Gy[Index(0, 0)] = c31
        Gy[Index(0, 2)] = s31
        Gy[Index(2, 0)] = -s31
        Gy[Index(2, 2)] = c31
        Axy = mc.mat_mat(Ax, Gy)

    # 3rd - Set element 10 to zero
    if A[Index(1, 0)] == 0:
        Axyz = Axy
        Gz = mc.eye(3)
    else:
        let r21 = math.sqrt(
            Axy[Index(1, 1)] * Axy[Index(1, 1)] + Axy[Index(1, 0)] * Axy[Index(1, 0)]
        )
        let c21 = Axy[Index(1, 1)] / r21
        let s21 = -Axy[Index(1, 0)] / r21
        Gz = mc.eye(3)
        Gz[Index(0, 0)] = c21
        Gz[Index(0, 1)] = -s21
        Gz[Index(1, 0)] = s21
        Gz[Index(1, 1)] = c21
        Axyz = mc.mat_mat(Axy, Gz)

    R = Axyz
    Q = mc.matT_matT(mc.matT_matT(Gz, Gy), Gx)


# fn KRt_from_P(P: Tensor[DType.float64]) -> (
#     Tensor[DType.float64],
#     Tensor[DType.float64],
#     Tensor[DType.float64],
# ):
#     var K = Tensor[DType.float64](3, 3)
#     var R = Tensor[DType.float64](3, 3)
#     var t = Tensor[DType.float64](3, 1)

#     # QR decomposition
#     rqGivens(P[0:3, 0:3], R, K)

#     # ensure that the diagonal is positive
#     if K[Index(2, 2)] < 0:
#         K = -K
#         R = -R
#     if K[Index(1, 1)] < 0:
#         let S = Tensor[DType.float64](3, 3)
#         S[Index(0, 0)] = 1
#         S[Index(1, 1)] = -1
#         S[Index(2, 2)] = 1
#         K = mc.mat_mat(K, S)
#         R = mc.mat_mat(S, R)
#     if K[Index(0, 0)] < 0:
#         let S = Tensor[DType.float64](3, 3)
#         S[Index(0, 0)] = -1
#         S[Index(1, 1)] = 1
#         S[Index(2, 2)] = 1
#         K = mc.mat_mat(K, S)
#         R = mc.mat_mat(S, R)

#     # ensure R determinant == 1
#     t = mc.solve(K, P[:, 3])

#     if mc.det(R) < 0:
#         R = -R
#         t = -t

#     K = K / K[Index(2, 2)]

#     return K, R, t
# function [K, R, t] = KRt_from_P(P)
# %% QR decomposition
# [K, R] = rqGivens(P(1:3, 1:3));

# %% ensure that the diagonal is positive
# if K(3, 3) < 0
#     K = -K;
#     R = -R;
# end
# if K(2, 2) < 0
#     S = [1  0  0
#          0 -1  0
#          0  0  1];
#     K = K * S;
#     R = S * R;
# end
# if K(1, 1) < 0
#     S = [-1  0  0
#           0  1  0
#           0  0  1];
#     K = K * S;
#     R = S * R;
# end

# %% ensure R determinant == 1
# t = linsolve(K, P(:, 4));

# if det(R) < 0
#     R = -R;
#     t = -t;
# end

# K = K ./ K(3, 3);

# end
