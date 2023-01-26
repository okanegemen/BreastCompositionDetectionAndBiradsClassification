def plot(H):
    train_epochs = [*range(len(H["train_loss"]))]
    val_epochs = [epoch for epoch in train_epochs if epoch%config.VALIDATE_PER_EPOCH==0]

    plt.plot(train_epochs,H["train_acc"])
    plt.plot(val_epochs,H["val_acc"])
    plt.savefig(config.PLOT_ACC_PATH)
    plt.clf()

    plt.plot(train_epochs,H["train_loss"])
    plt.plot(val_epochs,H["val_loss"])
    plt.savefig(config.PLOT_LOSS_PATH)
