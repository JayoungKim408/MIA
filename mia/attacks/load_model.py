import torch
from mia.synthesizers.utils import *

def load_model(model, data_name, train_data, meta_data, k=None, categorical_columns=tuple(), ordinal_columns=tuple(), dp=None, dp_type=None):

    model_path = './result/{}/model/{}'.format(data_name, model)

    if model == "ctgan":
        from mia.synthesizers.ctgan import CTGANSynthesizer as Synthesizer
        if dp:
            model_path += f'_{dp_type}'
    elif model == "octgan":
        from mia.synthesizers.octgan import OCTGANSynthesizer as Synthesizer
    elif model == "tvae":
        from mia.synthesizers.tvae import TVAESynthesizer as Synthesizer
    elif model == "tablegan":
        from mia.synthesizers.tablegan import TableganSynthesizer as Synthesizer
    else: # identity
        from mia.synthesizers.identity import IdentitySynthesizer as Synthesizer
        synthesizer = Synthesizer(data_name, meta_data)
        synthesizer.fit(train_data, categorical_columns, ordinal_columns)
        return synthesizer.sample(k)     

    synthesizer = Synthesizer(data_name, meta_data)

    if model == "ctgan" or model== "tvae" or model=='octgan':
        synthesizer.transformer = BGMTransformer()
        synthesizer.transformer.fit(train_data, synthesizer.dataset_name, categorical_columns, ordinal_columns)

    elif model == "tablegan":
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= train_data.shape[1]:
                side = i
                break

        synthesizer.transformer = TableganTransformer(side)
        synthesizer.transformer.fit(train_data, categorical_columns, ordinal_columns)

    else:
        synthesizer.transformer = GeneralTransformer()
        synthesizer.transformer.fit(train_data, synthesizer.dataset_name, categorical_columns, ordinal_columns, None)


    train = synthesizer.transformer.transform(train_data)
    if not model == "tablegan":
        data_dim = synthesizer.transformer.output_dim

    if model == "ctgan":
        from mia.synthesizers.ctgan import Cond, Generator
        synthesizer.cond_generator = Cond(train, synthesizer.transformer.output_info)
        synthesizer.generator = Generator(
                    synthesizer.embedding_dim + synthesizer.cond_generator.n_opt,
                    synthesizer.gen_dim,
                    data_dim).to(synthesizer.device)
        
    elif model == "octgan":
        from mia.synthesizers.octgan import Cond, Generator
        synthesizer.cond_generator = Cond(train, synthesizer.transformer.output_info)
        synthesizer.generator = Generator(
                    synthesizer.embedding_dim + synthesizer.cond_generator.n_opt,
                    synthesizer.gen_dim,
                    data_dim, rtol=1e-3, atol=1e-3).to(synthesizer.device)

    elif model == "tvae":
        from mia.synthesizers.tvae import Decoder
        synthesizer.decoder = Decoder(synthesizer.embedding_dim, synthesizer.compress_dims, data_dim).to(synthesizer.device)

    elif model == "tablegan":
        from mia.synthesizers.tablegan import Generator, determine_layers
        _, layers_G, _ = determine_layers(side, synthesizer.random_dim, synthesizer.num_channels)
        synthesizer.generator = Generator(synthesizer.transformer.meta, side, layers_G).to(synthesizer.device)

    # load weight
    model_path += ".pth"
    try:
        if model == "tvae":
            synthesizer.decoder.load_state_dict(torch.load(model_path)['G'])
        else:
            synthesizer.generator.load_state_dict(torch.load(model_path)['G'])
    except:
        print("Cannot find the model")
        return None

    return synthesizer

