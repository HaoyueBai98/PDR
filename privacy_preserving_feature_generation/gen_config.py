# d_model=params.hidden_size, encoder_layers=params.encoder_layers,
#                             decoder_layers=params.decoder_layers, encoder_attention_heads=params.encoder_heads,
#                             decoder_attention_heads=params.decoder_heads, decoder_ffn_dim=params.decoder_hidden,
#                             encoder_ffn_dim=params.encocer_hidden, dropout=params.dropout,
import argparse


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
                                                          'ap_omentum_ovary', 'german_credit',
                                                          'higgs', 'housing_boston', 'ionosphere',
                                                          'lymphography', 'messidor_features', 'openml_620',
                                                          'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                          'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
                                                          'openml_589', 'openml_607', 'openml_616', 'openml_618',
                                                          'openml_637'], default='airfoil')
    parser.add_argument('--mask_whole_op_p', type=float, default=0.0)
    parser.add_argument('--mask_op_p', type=float, default=0.0)
    parser.add_argument('--disorder_p', type=float, default=0.0)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--encoder_layers', type=int, default=1)
    parser.add_argument('--encoder_hidden_size', type=int, default=64)
    parser.add_argument('--encoder_emb_size', type=int, default=32)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden_size', type=int, default=200)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--decoder_hidden_size', type=int, default=64)

    parser.add_argument('--encoder_dropout', type=float, default=0)
    parser.add_argument('--mlp_dropout', type=float, default=0)
    parser.add_argument('--decoder_dropout', type=float, default=0)

    parser.add_argument('--gen_num', type=int, default=25)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--grad_bound', type=float, default=5.0)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--train_top_k', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()
    return args