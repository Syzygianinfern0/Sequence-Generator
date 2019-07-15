import tensorflow as tf
import numpy as np
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
import train
import time
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if __name__ == '__main__':
    print("\n\n1. Train \n2. Sample")
    c1 = input("\nChoose one : ")
    if c1 == '1':
        # Making the Agent
        print("\n\nJust press enter for default values")
        p1 = input("\nFile Path for training text (Def:'data/train_text/rj.txt') : ")
        p2 = input("Sequence Length (Def:100) : ")
        p3 = input("Batch Size (Def:64) : ")
        p1 = 'data/train_text/rj.txt' if p1 == '' else p1
        p2 = 100 if p2 == '' else int(p2)
        p3 = 64 if p3 == '' else int(p3)
        print("")
        agent = train.TextTrainer(file=p1,
                                  sequence_length=p2,
                                  batch_size=p3)

        # Making the Model
        print("\n\nJust press enter for default values")
        p1 = input("\nEmbedding Size (Def:256) : ")
        p2 = input("LSTM Units (Def:1024) : ")
        p3 = input("LSTM Layers (Def:1) : ")
        p4 = input("Learning Rate (Def:0.001) : ")
        p1 = 256 if p1 == '' else int(p1)
        p2 = 1024 if p2 == '' else int(p2)
        p3 = 1 if p3 == '' else int(p3)
        p4 = 0.001 if p4 == '' else int(p4)
        print("")
        model = agent.make_model(embedding_size=p1,
                                 lstm_units=p2,
                                 lstm_layers=p3,
                                 lr=p4)
        model.summary()

        # Training
        p5 = input("\n\nEpochs (Def:100) : ")
        p5 = 100 if p5 == '' else int(p5)
        c2 = input("\n\nDo you want to use Tensorboard to track? (y/n) : ")
        if c2.lower() == 'y':
            p6 = input(f"Enter Dir name (it will automatically be timestamped and saved under '.\\logs') "
                       f"\n (Def: .\\embedding_size-lstm_units-lstm_layers-timestamp\\)  : ")
            p6 = f"{p1}-{p2}-{p3}-{int(time.time())}" if p6 == '' else p6 + f"-{int(time.time())}"
            print(f"\n\n The logs will be saved at .\\logs\\{p6}")
            if not os.path.exists(f"logs\\{p6}"):
                os.makedirs(f"logs\\{p6}")
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs\\{p6}")

            agent.train(model=model,
                        epochs=p5,
                        callbacks=[tensorboard])
        elif c2.lower() == 'n':
            agent.train(model=model,
                        epochs=p5)

        c3 = input("Do you want to save your model? (y/n) : ")
        if c3.lower() == 'y':
            p7 = input(f"Enter File name (it will automatically be timestamped and saved under '.\\models') "
                       f"\n (Def: .\\models\\embedding_size-lstm_units-lstm_layers-epochs-timestamp.h5)  : ")
            p7 = f"{p1}-{p2}-{p3}-{p5}-{int(time.time())}.h5" if p7 == '' else p7 + f"-{int(time.time())}.h5"
            print(f"\n\n The logs will be saved at .\\models\\{p7}")
            if not os.path.exists(f".\\models"):
                os.makedirs(f".\\models")
            model.save_weights(filepath=f'models\\{p7}')

    elif c1 == '2':
        pass



