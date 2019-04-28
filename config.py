import torch


class Config:
    lr = 0.0001
    batch_size = 64
    DATA_PATH = "/content/dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnt_train = 150000
    cnt_test = 100
    size_image = (10, 50)

    # length_seq = 29
    length_seq = 55
    x, y = size_image
    input_dim = (3, x, y)

    kernel_1 = 3
    padding_1 = 1
    cnn1_dim = 64

    kernel_2 = 3
    padding_2 = 1
    cnn2_dim = 128

    kernel_3 = 3
    padding_3 = 1
    cnn3_dim = 256

    kernel_4 = 3
    padding_4 = 1
    cnn4_dim = 256

    kernel_5 = 3
    padding_5 = 1
    cnn5_dim = 512

    kernel_6 = 3
    padding_6 = 1
    cnn6_dim = 512

    kernel_7 = 2
    padding_7 = 0
    cnn7_dim = 512

    lstm_dim = (512, 512)
    # lstm_dim = (150, 512)
    output_dim = 37
    BLANK_LABEL = 0
    dropout = 0.5


    # size_image = (10, 50)
    #
    # # length_seq = 29
    # length_seq = 55
    # x, y = size_image
    # input_dim = (3, x, y)
    #
    # kernel_1 = 3
    # padding_1 = 1
    # cnn1_dim = 64
    #
    # kernel_2 = 3
    # padding_2 = 1
    # cnn2_dim = 128
    #
    # kernel_3 = 3
    # padding_3 = 1
    # cnn3_dim = 256
    #
    # kernel_4 = 3
    # padding_4 = 1
    # cnn4_dim = 256
    #
    # kernel_5 = 3
    # padding_5 = 1
    # cnn5_dim = 512
    #
    # kernel_6 = 3
    # padding_6 = 1
    # cnn6_dim = 512
    #
    # kernel_7 = 2
    # padding_7 = 0
    # cnn7_dim = 512
    #
    # lstm_dim = (512, 512)
    # # lstm_dim = (150, 512)
    # output_dim = 37
    # BLANK_LABEL = 0
    # dropout = 0.5
    
    print("#####################################     CONFIGURATION     #####################################")
    print("#\tCaptcha Recognition")
    print("#\tlr = ", lr)
    print("#\tbatch_size = ", batch_size)
    print("#\tnumber of samples train = ", cnt_train)
    print("#\tnumber of samples test = ", cnt_test)
    print("#\tImage size = ", (150, 50))
    print("#\tImage cut size = ", size_image )
    print("#\tLength seq image = ", length_seq)
    print("#\tNumber of layer CNN(conv) = ", 7)
    print("#\tOut channels CNN = ", cnn7_dim)
    print("#\tOut channels LSTM= ", lstm_dim[1])
    print("#\tDim target = ", output_dim)
    print("#################################################################################################")
