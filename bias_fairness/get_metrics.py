import csv

import nltk
import numpy as np
from classes import raw_classes

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

from composer.models.resnet import composer_resnet
from composer.trainer import Trainer
from composer.utils import S3ObjectStore

CKPT_BASE = 'ishana-resnet50-baseline/resnet50-imagenet-base-ej42/checkpoints/ep90-ba56250-rank0'
CKPT_MEDIUM = 'ishana-resnet50-medium/resnet50-imagenet-medium-ymk1/checkpoints/ep90-ba56250-rank0'
CKPT_MEDIUM_END = 'ishana-resnet50-medium/resnet50-imagenet-medium-ymk1/checkpoints/ep135-ba84375-rank0'
CKPT_PROGRES = 'ishana-resnet50-progres/resnet50-imagenet-progres-0ptr/checkpoints/ep90-ba56250-rank0'


def get_metrics():
    object_store = S3ObjectStore(
        bucket="mosaicml-internal-checkpoints-bert",  # The name of the cloud container (i.e. bucket) to use.
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    model = composer_resnet('resnet50')
    new_trainer = Trainer(
        model=model,  # max_duration="10ep",
        load_path='ishana-resnet50-baseline/resnet50-imagenet-base-78c3/checkpoints/ep90-ba56250-rank0',
        load_object_store=object_store,
    )

    for evald in new_trainer.state.eval_metrics.keys():
        with open('resnet50_medium_end_metrics.txt', 'w+') as f:
            for name, metric in new_trainer.state.eval_metrics[evald].items():
                print(f"\n{name.upper()}")
                # f.write(f"\n{name.upper()}")
                if name == 'ConfusionMatrix':
                    # print(list(metric.compute().cpu().numpy()))
                    for l in list(metric.compute().cpu().numpy()):
                        ls = np.array2string(l, separator=', ')
                        # f.write(f"\n{ls}")
                else:
                    l = metric.compute().cpu().numpy()
                    ls = np.array2string(l, separator=', ')
                    print(ls)
                    # f.write(f"\n{ls}")


def write_to_csv():
    object_store = S3ObjectStore(
        bucket="mosaicml-internal-checkpoints-bert",  # The name of the cloud container (i.e. bucket) to use.
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    model = composer_resnet('resnet50')
    new_trainer = Trainer(
        model=model,  # max_duration="10ep",
        load_path=CKPT_MEDIUM,
        load_object_store=object_store,
    )

    for evald in new_trainer.state.eval_metrics.keys():
        with open('resnet50_medium_confmat.csv', 'w+') as f:
            for name, metric in new_trainer.state.eval_metrics[evald].items():
                if name == 'ConfusionMatrix':
                    print(list(metric.compute().cpu().numpy()))
                    ls = []
                    for l in list(metric.compute().cpu().numpy()):
                        ls.append(np.array2string(l, separator=', '))

                    writer = csv.writer(f)
                    writer.writerows(ls)


def get_average_per_class():
    ALL_CKPTS_BASE = [
        'ishana-resnet50-baseline/resnet50-imagenet-base-78c3/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-ej42/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-i3j2/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-ovi8/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-icnq/checkpoints/ep90-ba56250-rank0'
    ]
    ALL_CKPTS_MEDIUM = [
        "ishana-resnet50-medium/resnet50-imagenet-medium-m8uw/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ihjp/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-9ffj/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ymk1/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-bugw/checkpoints/ep135-ba84375-rank0"
    ]
    ALL_CKPTS_MEDIUM_90 = [
        "ishana-resnet50-medium/resnet50-imagenet-medium-m8uw/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ihjp/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-9ffj/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ymk1/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-bugw/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_PROGRES = [
        "ishana-resnet50-progres/resnet50-imagenet-progres-u9a6/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-6651/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-f9x8/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-51fu/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-0ptr/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_AUGMIX = [
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-f0ir/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-troy/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-sa04/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-m029/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-cjfw/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_MILD = [
        "ishana-resnet50-mild/resnet50-imagenet-mild-fdjx/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-cjzm/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-2mqo/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-6br9/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-6qus/checkpoints/ep36-ba22500-rank0"
    ]
    ALL_CKPTS_EMA = [
        "ishana-resnet50-ema/resnet50-imagenet-ema-wk83/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-ema/resnet50-imagenet-ema-bpc5/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-ema/resnet50-imagenet-ema-debx/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-ema/resnet50-imagenet-ema-tmmg/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-ema/resnet50-imagenet-ema-bvry/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_BLURPOOL = [
        "ishana-resnet50-blurpool/resnet50-imagenet-blurpool-m9rg/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-blurpool/resnet50-imagenet-blurpool-nic9/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-blurpool/resnet50-imagenet-blurpool-uvlz/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-blurpool/resnet50-imagenet-blurpool-le4a/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-blurpool/resnet50-imagenet-blurpool-zlpm/checkpoints/ep90-ba56250-rank0"
    ]

    object_store = S3ObjectStore(
        bucket="mosaicml-internal-checkpoints-bert",  # The name of the cloud container (i.e. bucket) to use.
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    model = composer_resnet('resnet50')

    per_class_accs = []
    for ckpt in ALL_CKPTS_MEDIUM_90:
        new_trainer = Trainer(
            model=model,
            load_path=ckpt,
            load_object_store=object_store,
        )
        for evald in new_trainer.state.eval_metrics.keys():
            for name, metric in new_trainer.state.eval_metrics[evald].items():
                if name == 'PerClassAccuracy':
                    l = metric.compute().cpu().numpy()
                    per_class_accs.append(list(l))
        new_trainer.close()
    with open('resnet50_medium_90_average_metrics.txt', 'w+') as f:
        average = np.average(per_class_accs, axis=0)
        print(average)
        f.write(np.array2string(average, separator=', '))


def get_words():
    """Get class names of worst performing classes."""
    from results import pca_augmix_average, pca_base_average, pca_medium_average, pca_mild_average, pca_progres_average
    from super_classes import super_classes
    words = []
    sc = {}
    for i, (_, a) in enumerate(reversed(sorted(zip(pca_base_average, list(raw_classes.values())))[:100])):
        # print(f"{c}: {a}")
        words.append(a)
        if super_classes[i] not in sc:
            sc[super_classes[i]] = 0
        sc[super_classes[i]] += 1  #count number of times superclass appears

    sc_sorted = [k for k, _ in reversed(sorted(sc.items(), key=lambda item: item[1]))]
    print(sc_sorted)
    return sc_sorted
    # print(f"\nClass {classes[list(l).index(min(l))]}: {min(l)}")


def get_synsets():
    for word in get_words():
        for w in word.split(','):
            try:
                syn = wordnet.synsets(w)[0]
                print("\nname :  ", syn.name())
                print("hypernyms :  ", syn.hypernyms()[0].name())
            except Exception:
                print(f'\nfailed to find synset for {word}')


def get_error_per_class():
    ALL_CKPTS_BASE = [
        'ishana-resnet50-baseline/resnet50-imagenet-base-78c3/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-ej42/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-i3j2/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-ovi8/checkpoints/ep90-ba56250-rank0',
        'ishana-resnet50-baseline/resnet50-imagenet-base-icnq/checkpoints/ep90-ba56250-rank0'
    ]
    ALL_CKPTS_MEDIUM = [
        "ishana-resnet50-medium/resnet50-imagenet-medium-m8uw/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ihjp/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-9ffj/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-ymk1/checkpoints/ep135-ba84375-rank0",
        "ishana-resnet50-medium/resnet50-imagenet-medium-bugw/checkpoints/ep135-ba84375-rank0"
    ]
    ALL_CKPTS_PROGRES = [
        "ishana-resnet50-progres/resnet50-imagenet-progres-u9a6/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-6651/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-f9x8/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-51fu/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-progres/resnet50-imagenet-progres-0ptr/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_AUGMIX = [
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-f0ir/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-troy/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-sa04/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-m029/checkpoints/ep90-ba56250-rank0",
        "ishana-resnet50-augmix/resnet50-imagenet-augmix-cjfw/checkpoints/ep90-ba56250-rank0"
    ]
    ALL_CKPTS_MILD = [
        "ishana-resnet50-mild/resnet50-imagenet-mild-fdjx/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-cjzm/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-2mqo/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-6br9/checkpoints/ep36-ba22500-rank0",
        "ishana-resnet50-mild/resnet50-imagenet-mild-6qus/checkpoints/ep36-ba22500-rank0"
    ]

    object_store = S3ObjectStore(
        bucket="mosaicml-internal-checkpoints-bert",  # The name of the cloud container (i.e. bucket) to use.
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    model = composer_resnet('resnet50')

    per_class_accs = []
    for ckpt in ALL_CKPTS_BASE:
        new_trainer = Trainer(
            model=model,
            load_path=ckpt,
            load_object_store=object_store,
        )
        for evald in new_trainer.state.eval_metrics.keys():
            for name, metric in new_trainer.state.eval_metrics[evald].items():
                if name == 'PerClassAccuracy':
                    l = metric.compute().cpu().numpy()
                    per_class_accs.append(list(l))
        new_trainer.close()
    with open('resnet50_base_error_metrics.txt', 'w+') as f:
        average = np.min(per_class_accs, axis=0)
        print(average)
        f.write(np.array2string(average, separator=', '))

        average = np.max(per_class_accs, axis=0)
        print(average)
        f.write(np.array2string(average, separator=', '))


# get_metrics()
get_average_per_class()
# get_error_per_class()
# write_to_csv()
# print(get_words())
# get_synsets()
