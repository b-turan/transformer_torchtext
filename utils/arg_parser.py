import argparse

def create_parser():
    """
    Arguments devided into Trainer, Model and Program specific arguments.
    """
    parser = argparse.ArgumentParser(description="Transformer Model for Translation")
    # Trainer args  (gpus, epochs etc.)
    parser.add_argument("-d", "--dataset", type=str, metavar="", help="Choice of Dataset", default="multi30k")
    parser.add_argument("-g", "--gpus", type=int, metavar="", help="Number of GPUS, (None for CPU)", default=1)
    parser.add_argument("--batch_size", type=int, metavar="", help="Batch Size", default=128)
    parser.add_argument("-lr","--learning_rate", type=float, metavar="", help="Initial Learning Rate", default= 5e-4)
    parser.add_argument("-e","--epochs", type=int, metavar="", help="Number of Epochs", default= 2)
    parser.add_argument("--clip", type=int, metavar="", help="Gradient Clipping", default= 1)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)

    # Model specific arguments
    parser.add_argument("--hid_dim", type=int, metavar="", help="Hidden Dimension", default=256)
    parser.add_argument("--enc_layers", type=int, metavar="", help="Number of Encoder Layers", default=3)
    parser.add_argument("--dec_layers", type=int, metavar="", help="Number of Decoder Layers", default=3)
    parser.add_argument("--enc_heads", type=int, metavar="", help="Number of Encoder Heads", default=8)
    parser.add_argument("--dec_heads", type=int, metavar="", help="Number of Decoder Heads", default=8)
    parser.add_argument("--enc_pf_dim", type=int, metavar="", help="...TODO...", default=512)
    parser.add_argument("--dec_pf_dim", type=int, metavar="", help="...TODO...", default=512)
    parser.add_argument("--enc_dropout", type=float, metavar="", help="Dropout Rate of Encoder Layers", default=0.1)
    parser.add_argument("--dec_dropout", type=float, metavar="", help="Dropout Rate of Decoder Layers", default=0.1)

    # Program arguments (data_path, save_dir, etc.)
    parser.add_argument("--seed", type=int, metavar="", help="Seed Choice", default=1234)

    return parser
