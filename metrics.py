from matplotlib import pyplot
import numpy as np

# plot learning curves
def summarize_diagnostics(history, args):
	# plot loss
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.xticks(np.arange(0, len(history.history['loss']), 2.0))
    pyplot.yticks(np.arange(0, 1.1, 0.1))
    pyplot.legend(('train', 'validation'),loc='upper right')

	# plot accuracy
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.xticks(np.arange(0, len(history.history['loss']), 2.0))
    pyplot.yticks(np.arange(0, 1.1, 0.1))
    pyplot.legend(('train', 'validation'),loc='upper right')
	# save plot to file
    pyplot.savefig(args.id + '_cats_dogs.png')
    pyplot.close()
