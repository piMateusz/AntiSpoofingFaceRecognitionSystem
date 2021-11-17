import matplotlib.pyplot as plt


def plot_img_and_cropped(img, cropped):
    fig = plt.figure(figsize=(50, 50))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Full image", fontsize=20)

    fig.add_subplot(1, 2, 2)
    plt.imshow(cropped)
    plt.axis('off')
    plt.title("Cropped image", fontsize=20)

    plt.show()


def plot_model_evaluation(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.show()
