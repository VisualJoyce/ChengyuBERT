import importlib
import os

MODEL_REGISTRY = {}


def build_model(opts):
    if opts.model.startswith('chengyubert-emb'):
        ModelCls = MODEL_REGISTRY['chengyubert-emb']
        opts.evaluate_embedding = True
    elif opts.model.startswith('chengyubert-ns'):
        ModelCls = MODEL_REGISTRY['chengyubert-ns']
        opts.evaluate_embedding = True
    elif opts.model.startswith('chengyubert-contrastive'):
        ModelCls = MODEL_REGISTRY['chengyubert-contrastive']
        opts.evaluate_embedding = True
    else:
        ModelCls = MODEL_REGISTRY[opts.model]
    return ModelCls.from_pretrained(opts.pretrained_model_name_or_path,
                                    len_idiom_vocab=opts.len_idiom_vocab,
                                    model_name=opts.model)


def register_model(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('chengyubert.models.' + model_name)
