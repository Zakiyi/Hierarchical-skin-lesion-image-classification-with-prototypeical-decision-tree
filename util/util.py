import base64
import warnings
import numpy as np
from io import BytesIO
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics._classification import _check_targets
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
from sklearn.metrics import balanced_accuracy_score


def generate_wandb_name(backbone, dataset, hierarchy, metric_guided, loss, **kwargs):
    if hierarchy is not None:
        rname = 'tree-model'
    else:
        rname = 'flat-model'

    if metric_guided:
        rname += '-guided'

    rname += f'-{backbone}-{dataset}'
    if 'SoftTreeSupLoss' in loss:
        rname += f'-{loss[0]}-xw{kwargs["xent_weight"]}-xwe{kwargs["xent_weight_end"]}-p{kwargs["xent_weight_power"]}' \
                 f'-tsw{kwargs["tree_supervision_weight"]}-tswe{kwargs["tree_supervision_weight_end"]}-p{kwargs["tree_supervision_weight_power"]}'

    return rname


def classification_report(y_true, y_pred, *, labels=None, target_names=None,
                          sample_weight=None, digits=2, output_dict=False,
                          zero_division="warn"):

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True

    # labelled micro average
    micro_is_accuracy = ((y_type == 'multiclass' or y_type == 'binary') and
                         (not labels_given or
                          (set(labels) == set(unique_labels(y_true, y_pred)))))

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ['%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight,
                                                  zero_division=zero_division)
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted', 'bacc')

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers,
                                          [i.item() for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += '\n'

    # compute all applicable averages
    for average in average_options:
        if average.startswith('micro') and micro_is_accuracy:
            line_heading = 'accuracy'
        else:
            line_heading = average + ' avg'

        # compute averages with specified averaging method
        if line_heading == 'bacc avg':
            avg = [bacc, bacc, bacc, np.sum(s)]
        else:
            avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels,
                average=average, sample_weight=sample_weight,
                zero_division=zero_division)
            avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(
                zip(headers, [i.item() for i in avg]))
        else:
            if line_heading == 'accuracy':
                row_fmt_accuracy = '{:>{width}s} ' + \
                        ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
                        ' {:>9}\n'
                report += row_fmt_accuracy.format(line_heading, '', '',
                                                  *avg[2:], width=width,
                                                  digits=digits)
            elif line_heading == 'bacc avg':
                row_fmt_accuracy = '{:>{width}s} ' + \
                        ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
                        ' {:>9}\n'
                report += row_fmt_accuracy.format(line_heading, '', '',
                                                  *avg[2:], width=width,
                                                  digits=digits)
            else:
                report += row_fmt.format(line_heading, *avg,
                                         width=width, digits=digits)

    if output_dict:
        if 'accuracy' in report_dict.keys():
            report_dict['accuracy'] = report_dict['accuracy']['precision']
        return report_dict
    else:
        return report


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    return im


def array_to_babse64(arr):
    im_pil = Image.fromarray(arr)
    im_pil.resize( (256, 256) )
    buffered = BytesIO()
    im_pil.save(buffered, format="JPEG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/jpeg;base64," + base64.b64encode(img_byte).decode()
    return img_str


def to_string(num_arr):
    return ', '.join([str(i) for i in num_arr])


def convert_color(c):
    return 'rgb({},{},{})'.format(int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))


def convert_color_hex(v):
    c = cm.jet(v)
    return "#{:02x}{:02x}{:02x}".format(int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))


def get_prob_table(probs_dict, i2l2, l2i):
    nrows = len(i2l2)
    ncols = 3
    labels = []
    colors = []

    lasts = ['', '', '']
    for l2 in i2l2:
        parts = l2.split(':')
        for c in range(3):
            l = parts[c]
            ind = l2i[c][':'.join(parts[:c + 1])]
            v = probs_dict[c][ind]
            color = convert_color_hex(v)
            colors.append(color)
            if l != lasts[c]:
                labels.append('{} ({:.4f})'.format(l, v))
                lasts[c] = l
            else:
                labels.append('')

    return {
        'rows': nrows,
        'cols': ncols,
        'labels': labels,
        'colors': colors
    }


def prob_table_to_image(prob_table, output_file, IMG_W=960, IMG_H=1400):
    CELL_H = IMG_H / prob_table['rows']
    CELL_W = IMG_W / prob_table['cols']
    img = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("fonts/Verdana.ttf", 11)

    ind = 0
    for r in range(prob_table['rows']):
        for c in range(prob_table['cols']):
            pos_x = c * CELL_W
            pos_y = r * CELL_H
            draw.rectangle([(pos_x, pos_y), (pos_x + CELL_W, pos_y + CELL_H)], prob_table['colors'][ind])
            draw.text((pos_x + 5, pos_y + 5), prob_table['labels'][ind], font=fnt)
            ind += 1

    # save
    img.save(output_file)


def basic_stats(data, level=0):
    actuals = []
    predictions = []
    for k in data.keys():
        item = data[k]
        actuals.append(int(item['actuals'].split(',')[level]))
        pred = item['pred_l{}'.format(level)].split(',')
        pred = np.asarray(pred, dtype=np.float32)
        predictions.append(np.argmax(pred))

    print('\n=== Basic stats level {} ==='.format(level))
    print('Confusion matrix:')
    print(confusion_matrix(actuals, predictions))
    print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
    print('Accuracy score: %f' % accuracy_score(actuals, predictions))


def my_classification_report(data, i2l_dict, level=0, output_dict=True):
    actuals = []
    predictions = []
    i2l = i2l_dict[level]
    for k in data.keys():
        item = data[k]
        actuals.append(i2l[int(item['actuals'].split(',')[level])])
        pred = item['pred_l{}'.format(level)].split(',')
        pred = np.asarray(pred, dtype=np.float32)
        predictions.append(i2l[np.argmax(pred)])
    # bacc = balanced_accuracy_score(actuals, predictions)
    # result = {'label': actuals, 'pred': predictions}
    return classification_report(actuals, predictions, output_dict=output_dict, digits=4)


def topk_acc(data, k=3, level=1):
    num_correct = 0
    for key in data.keys():
        item = data[key]
        actual = int(item['actuals'].split(',')[level])
        pred = item['pred_l{}'.format(level)].split(',')
        pred = np.asarray(pred, dtype=np.float32)
        idx = pred.argsort()[::-1][:k]
        if actual in idx:
            num_correct += 1
        else:
            # TODO:
            pass

    total = len(data.keys())
    acc = 100.0 * num_correct / total
    print('\n=== Top {} accuracy for level {} ==='.format(k, level))
    print('Total = {}, correct = {}, accuracy = {}'.format(total, num_correct, acc))


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, acc=False):
        self.reset()
        self.is_acc = acc
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.is_acc:
            self.sum += val
        else:
            self.sum += val * n

        self.count += n
        self.avg = self.sum / self.count


def get_l1_label(l2):
    return ':'.join(l2.split(':')[0:2])


def get_l0_label(l1):
    return l1.split(':')[0]


def covert_preds(data, l02i, l12i, l22i, i2l0, i2l1, i2l2):
    # map between level1 to level2
    l1_l2 = {}
    for l1 in i2l1:
        l1_l2[l1] = []
        for l2 in i2l2:
            if get_l1_label(l2) == l1:
                l1_l2[l1].append(l22i[l2])

    # map between level0 to level1
    l0_l1 = {}
    for l0 in i2l0:
        l0_l1[l0] = []
        for l1 in i2l1:
            if get_l0_label(l1) == l0:
                l0_l1[l0].append(l12i[l1])

    # calculate probs for l0 and l1
    for k in data.keys():
        item = data[k]
        pred_l2 = np.asarray(item['pred_l2'].split(','), dtype=np.float32)
        pred_l1 = []
        for l1 in i2l1:
            indexes = l1_l2[l1]
            arr = pred_l2[indexes]
            pred_l1.append(np.sum(arr))
        pred_l1 = [str(v) for v in pred_l1]
        data[k]['pred_l1'] = ', '.join(pred_l1)

        pred_l0 = []
        pred_l1 = np.asarray(pred_l1, dtype=np.float32)
        for l0 in i2l0:
            indexes = l0_l1[l0]
            arr = pred_l1[indexes]
            pred_l0.append(np.sum(arr))
        pred_l0 = [str(v) for v in pred_l0]
        data[k]['pred_l0'] = ', '.join(pred_l0)

        #actuals
        l2 = i2l2[int(item['actuals'])]
        l1 = get_l1_label(l2)
        l0 = get_l0_label(l1)
        data[k]['actuals'] = '{}, {}, {}'.format(l02i[l0], l12i[l1], item['actuals'])

    return data