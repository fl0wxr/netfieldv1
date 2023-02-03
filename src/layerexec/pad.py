class pad:

    def __init__(self, tm_):

        self.tm_ = tm_

    def output(self, T, p):
        """
        Description:
            The input is a fourth rank tensor but each training example corresponds to a third rank tensor, which is replaced by its padded version.

        Inputs:
            <T>: ndarray with shape (examples, height, width, channels), this tensor will be mapped to include a padding around each of the channel and example.
            <p>: int, padding size

        Returns:
            <T_pad>: ndarray with shape (examples, height+2*p, width+2*p, channels), this is the padded version of <T>.
        """

        T_pad = self.tm_.tm.zeros((T.shape[0], T.shape[1]+2*p, T.shape[2]+2*p, T.shape[3]))
        T_pad[:, p:T_pad.shape[1]-p, p:T_pad.shape[2]-p, :] = T

        return T_pad
