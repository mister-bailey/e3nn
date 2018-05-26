# pylint: disable=C,R,E1101
from functools import partial
import torch
from se3_cnn import SE3BNConvolution, SE3Convolution, SE3GNConvolution
from se3_cnn.non_linearities import ScalarActivation
from se3_cnn.dropout import SE3Dropout
from se3_cnn import basis_kernels


class GatedBlock(torch.nn.Module):
    def __init__(self,
                 repr_in, repr_out, size, radial_window=basis_kernels.gaussian_window_fct_convenience_wrapper,  # kernel params
                 activation=(None, None), stride=1, padding=0, capsule_dropout_p=None,  # conv/nonlinearity/dropout params
                 normalization=None, batch_norm_momentum=0.1,  # batch norm params
                 verbose=False):
        '''
        :param repr_in: tuple with multiplicities of repr. (1, 3, 5, ..., 15)
        :param repr_out: same but for the output
        :param int size: the filters are cubes of dimension = size x size x size
        :param radial_window: radial window function
        :param activation: (scalar activation, gate activation) which are functions like torch.nn.functional.relu or None
        :param int stride: stride of the convolution (for torch.nn.functional.conv3d)
        :param int padding: padding of the convolution (for torch.nn.functional.conv3d)
        :param float capsule_dropout_p: dropout probability
        :param str normalization: "batch", "group", "instance" or None
        :param float batch_norm_momentum: batch normalization momentum (ignored if no batch normalization)
        '''
        super().__init__()

        if type(activation) is tuple:
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation

        self.repr_out = repr_out

        Rs_in = [(m, l) for l, m in enumerate(repr_in)]
        Rs_out_with_gate = [(m, l) for l, m in enumerate(repr_out)]

        if (scalar_activation is not None and repr_out[0] > 0):
            self.scalar_act = ScalarActivation([(repr_out[0], scalar_activation)])
        else:
            self.scalar_act = None

        n_non_scalar = sum(repr_out[1:])
        if gate_activation is not None and n_non_scalar > 0:
            Rs_out_with_gate.append((n_non_scalar, 0))  # concatenate scalar gate capsules after normal capsules
            self.gate_act = ScalarActivation([(n_non_scalar, gate_activation)])
        else:
            self.gate_act = None

        if normalization == None:
            Convolution = SE3Convolution
        elif normalization == "batch":
            Convolution = partial(SE3BNConvolution, momentum=batch_norm_momentum)
        elif normalization == "group":
            Convolution = SE3GNConvolution
        elif normalization == "instance":
            Convolution = partial(SE3GNConvolution, Rs_gn=[(1, 2 * n + 1) for n, mul in enumerate(repr_in) for _ in range(mul)])
        else:
            raise NotImplementedError('normalization mode unknown')

        self.conv = Convolution(
            Rs_in=Rs_in,
            Rs_out=Rs_out_with_gate,
            size=size,
            radial_window=radial_window,
            stride=stride,
            padding=padding,
            verbose=verbose,
        )

        self.dropout = None
        if capsule_dropout_p is not None:
            Rs_out_without_gate = [(mul, 2 * n + 1) for n, mul in enumerate(repr_out)]  # Rs_out without gates
            self.dropout = SE3Dropout(Rs_out_without_gate, capsule_dropout_p)

    def forward(self, x):  # pylint: disable=W

        # convolution
        y = self.conv(x)

        if self.scalar_act is None and self.gate_act is None:
            z = y
        else:
            nbatch = y.size(0)
            nx = y.size(2)
            ny = y.size(3)
            nz = y.size(4)

            size_out = sum(mul * (2 * n + 1) for n, mul in enumerate(self.repr_out))

            if self.gate_act is not None:
                g = y[:, size_out:]
                g = self.gate_act(g)
                begin_g = 0  # index of first scalar gate capsule

            z = y.new_empty((y.size(0), size_out, y.size(2), y.size(3), y.size(4)))
            begin_y = 0  # index of first capsule

            for n, mul in enumerate(self.repr_out):
                if mul == 0:
                    continue
                dim = 2 * n + 1

                # crop out capsules of order n
                field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, x, y, z]

                if n == 0:
                    # Scalar activation
                    if self.scalar_act is not None:
                        field = self.scalar_act(field_y)
                    else:
                        field = field_y
                else:
                    if self.gate_act is not None:
                        # reshape channels in capsules and capsule entries
                        field_y = field_y.contiguous()
                        field_y = field_y.view(nbatch, mul, dim, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                        # crop out corresponding scalar gates
                        field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, x, y, z]
                        begin_g += mul
                        # reshape channels for broadcasting
                        field_g = field_g.contiguous()
                        field_g = field_g.view(nbatch, mul, 1, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                        # scale non-scalar capsules by gate values
                        field = field_y * field_g  # [batch, feature, repr, x, y, z]
                        field = field.view(nbatch, mul * dim, nx, ny, nz)  # [batch, feature * repr, x, y, z]
                        del field_g
                    else:
                        field = field_y

                z[:, begin_y: begin_y + mul * dim] = field
                begin_y += mul * dim
                del field

        # dropout
        if self.dropout is not None:
            z = self.dropout(z)

        return z
