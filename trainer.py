
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
import src.utility as utility
import src.models as models
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import Callback
import random
import tensorflow as tf
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return tf.keras.callbacks.LearningRateScheduler(schedule)
def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy.
    """
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.savefig("")

def time_shift(spec):
    if np.random.rand() <= 0.5:
        shift = random.randint(-5, 5)
        if shift > 0:
            padded = np.pad(spec, ((0, 0), (0, 0), (shift, 0), (0, 0)), "constant")
            return padded[:, :, :-shift, :]
        elif shift < 0:
            padded = np.pad(spec, ((0, 0), (0, 0), (0, -shift), (0, 0)), "constant")
            return padded[:, :, -shift:, :]
        else:
            return spec
    else:
        return spec
def time_masking(S):
    # time_masking
    if np.random.rand()<=0.5:
        num_time_bins = S.shape[1]
        time_mask_length = np.random.randint(low=0, high=num_time_bins // 5+1)
        t0 = np.random.randint(low=0, high=num_time_bins - time_mask_length)
        S[:, t0:t0 + time_mask_length] = 0
    return S
def add_noise(S):
    if np.random.rand()<=0.5:
        noise = np.random.randn(*S.shape)
        S = S + 0.0005 * noise
    return S
def f_masking(S):
    if np.random.rand()<=0.5:
        num_mel_bins = S.shape[0]
        freq_mask_length = np.random.randint(low=0, high=num_mel_bins // 5+1)
        f0 = np.random.randint(low=0, high=num_mel_bins - freq_mask_length)
        S[f0:f0 + freq_mask_length, :] = 0
    return S
def data_generator(X, Y, batch_size, augment_functions=[], shuffle=True):
    num_samples = X.shape[0]
    if shuffle:
        indices = np.random.permutation(num_samples)
    else:
        indices = np.arange(num_samples)
    
    while True:  # Loop forever so the generator never terminates
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = X[batch_indices]
            batch_y = Y[batch_indices]

            # Apply augmentations
            for augment_func in augment_functions:
                batch_x = augment_func(batch_x)
            yield batch_x, batch_y

        
def train_model(nb_classes=20,
                slice_length=911,
                artist_folder='artist20',
                song_folder='song_split_data',
                plots=True,
                train=True,
                load_checkpoint=False,
                save_metrics=True,
                save_metrics_folder='metrics',
                save_weights_folder='weights',
                batch_size=16,
                nb_epochs=100,
                early_stop=10,
                lr=0.0001,
                album_split=True,
                random_states=42):
    """
    Main function for training the model and testing
    """

    weights = os.path.join(save_weights_folder, str(nb_classes) +
                           '_' + str(slice_length) + '_' + str(random_states)+'no_aug')
    os.makedirs(save_weights_folder, exist_ok=True)
    os.makedirs(save_metrics_folder, exist_ok=True)

    print("Loading dataset...")

    if not album_split:
        # song split
        Y_train, X_train, S_train, \
        Y_val, X_val, S_val = \
            utility.load_dataset_song_split(song_folder_name=song_folder,
                                            artist_folder=artist_folder,
                                            nb_classes=nb_classes,
                                            random_state=random_states)
        X_testing, song_length, song_name= utility.load_test_set(song_folder_name='song_split_test_data')
    else:
        Y_train, X_train, S_train,\
        Y_val, X_val, S_val = \
            utility.load_dataset_album_split(song_folder_name=song_folder,
                                             artist_folder=artist_folder,
                                             nb_classes=nb_classes,
                                             random_state=random_states)
        X_testing, song_length, song_name= utility.load_test_set(song_folder_name='song_split_test_data')

    print("Loaded and split dataset. Slicing songs...")
    X_train, Y_train, S_train = utility.slice_train_songs(X_train, Y_train, S_train,
                                                        length=slice_length)
    X_val, Y_val, S_val = utility.slice_val_songs(X_val, Y_val, S_val,
                                            length=slice_length)
    # Encode the target vectors into one-hot encoded vectors
    Y_train, le, enc = utility.encode_labels(Y_train)
    Y_val, le, enc = utility.encode_labels(Y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))
    print("X_val.shape",X_val.shape)
    X_testing,song_slices= utility.slice_test_songs(X_testing, length= slice_length)
    # print("Training set label counts:", np.unique(Y_train, return_counts=True))

   
    X_testing = X_testing.reshape(X_testing.shape + (1,))

    # build the model
    model = models.CRNN2D(X_testing.shape, nb_classes=20)
    print("X_testing.shape",X_testing.shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    # model2 = models.CRNN2D(X_testing.shape, nb_classes=20)
    # model2.compile(loss='categorical_crossentropy',
    #               optimizer=Adam(lr=lr),
    #               metrics=['accuracy'])

    # model3 = models.CRNN2D(X_testing.shape, nb_classes=20)
    # model3.compile(loss='categorical_crossentropy',
    #               optimizer=Adam(lr=lr),
    #               metrics=['accuracy'])              
    # model.summary()

    # Initialize weights using checkpoint if it exists
    if load_checkpoint:
        print("Looking for previous weights...")
        if isfile(weights):
            print(f'load {weights}')
            model.load_weights("/home/fundwotsai/music-artist-classification-crnn/weights_album_split/20_128_21no_aug")
            # model2.load_weights("/home/fundwotsai/music-artist-classification-crnn/weights_album_split/20_132_4")
            # model3.load_weights("/home/fundwotsai/music-artist-classification-crnn/weights_album_split/20_132_40")
            # model.summary()
        else:
            print('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')

    checkpointer = ModelCheckpoint(filepath=weights,
                                   verbose=1,
                                   save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                                 patience=early_stop, verbose=0, mode='auto')

    import keras.backend as K

    # ... (Your model definition and compilation here)

    # Initialize all variables
    # Train the model
    if train:
       
        val_gen = data_generator(X_val, Y_val, batch_size, shuffle=False)

        print("val_gen ",val_gen)
        for epoch in range(nb_epochs):
            Y_train, X_train, S_train,\
            _, _,_ = \
                utility.load_dataset_album_split(song_folder_name=song_folder,
                                                artist_folder=artist_folder,
                                                nb_classes=nb_classes,
                                                random_state=random_states)
            X_train, Y_train, S_train = utility.slice_train_songs(X_train, Y_train, S_train,
                                                    length=slice_length)
            Y_train, le, enc = utility.encode_labels(Y_train)
            X_train = X_train.reshape(X_train.shape + (1,))
            train_gen = data_generator(X_train, Y_train, batch_size,augment_functions=[time_shift,time_masking,add_noise,f_masking])
            # lr_scheduler = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
            history = model.fit_generator(train_gen, 
                                        steps_per_epoch=len(X_train) // batch_size, 
                                        epochs=1, 
                                        verbose=1, 
                                        validation_data=val_gen, 
                                        validation_steps=len(X_val) // batch_size, 
                                        callbacks=[checkpointer, earlystopper])
            val_loss, val_accuracy = model.evaluate_generator(generator=val_gen, 
                                                   steps=len(X_val) // batch_size)
            print('Validation generator Loss:', val_loss)
            print('Validation generator Accuracy:', val_accuracy)
            # plot_training_history(history, f'training_{slice_length}_history_plot.png')
    # val_gen = data_generator(X_val, Y_val, batch_size, shuffle=False)
    # val_loss, val_accuracy = model.evaluate_generator(generator=val_gen, 
    #                                                steps=len(X_val) // batch_size)
    # print('Validation generator Loss:', val_loss)
    # print('Validation generator Accuracy:', val_accuracy)
    # val_loss2, val_accuracy2 = model2.evaluate_generator(generator=val_gen, 
    #                                                steps=len(X_val) // batch_size)
    # print('Validation generator Loss2:', val_loss2)
    # print('Validation generator Accuracy2:', val_accuracy2)
    # val_loss3, val_accuracy3 = model3.evaluate_generator(generator=val_gen, 
    #                                                steps=len(X_val) // batch_size)
    # print('Validation generator Loss3:', val_loss3)
    # print('Validation generator Accuracy3', val_accuracy3)
    # # model.load_weights(weights)
    filename = os.path.join(save_metrics_folder, str(nb_classes) + '_'
                            + str(slice_length)
                            + '_' + str(random_states) +"no_aug"+ '.txt')
    print("filename",filename)
    y_score = model.predict_proba(X_testing)
    # y_score2 = model2.predict_proba(X_testing)
    # y_score3 = model3.predict_proba(X_testing)
    # y_score = (y_score + y_score2 + y_score3)/3
    # # print(type(y_score))
    # # print(y_score.shape)
    
    # current_length = 0
    # pool_top_3 = []
    # for i in range(len(song_length)):
    #     # print("song_length[i]",song_length[i])
    #     score = np.sum(y_score[current_length : current_length + song_slices[i], : ],axis=0)
    #     # print("score shape:",score.shape)
    #     y_top3_indices = score.argsort()[-3:][::-1]
    #     pool_top_3.append(y_top3_indices)
    #     current_length = current_length + song_slices[i]
    #     # print("current_length" ,current_length)
    # artist_list = ['aerosmith', 'beatles' ,'creedence_clearwater_revival' ,'cure',\
    # 'dave_matthews_band' ,'depeche_mode' ,'fleetwood_mac', 'garth_brooks',\
    # 'green_day' ,'led_zeppelin', 'madonna' ,'metallica' ,'prince', 'queen',\
    # 'radiohead', 'roxette', 'steely_dan', 'suzanne_vega' ,'tori_amos' ,'u2']
    # with open("test_myvoice.csv", 'w') as f:
    #     for i in range(len(song_length)):
    #         f.write('{},{},{},{}\n'.format(song_name[i],artist_list[pool_top_3[i][0]],artist_list[pool_top_3[i][1]],artist_list[pool_top_3[i][2]]))
    #     f.close

    score = model.evaluate(X_val, Y_val, verbose=0)
    y_score = model.predict_proba(X_val)
    print(score)
    # Calculate confusion matrix
    y_predict = np.argmax(y_score, axis=1)
    y_true = np.argmax(Y_val, axis=1)

    cm = confusion_matrix(y_true, y_predict)

    # Plot the confusion matrix
    class_names = np.arange(nb_classes)
    class_names_original = le.inverse_transform(class_names)
    plt.figure(figsize=(14, 14))
    
    utility.plot_confusion_matrix(cm, classes=class_names_original,
                                  normalize=True,
                                  title='Confusion matrix with normalization')
    if save_metrics:
        plt.savefig(filename + '.png', bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(14, 14))
    # Print out metrics
    print('Test score/loss:', score[0])
    print('Test accuracy:', score[1])
    print('\nTest results on each slice:')
    scores = classification_report(y_true, y_predict,
                                   target_names=class_names_original)
    scores_dict = classification_report(y_true, y_predict,
                                        target_names=class_names_original,
                                        output_dict=True)
    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y_val, axis=1)
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
    print(f'Manually calculated accuracy: {accuracy * 100:.2f}%')

    # Predict artist using pooling methodology
    pooling_scores, pooled_scores_dict = \
        utility.predict_artist(model, X_val, Y_val, S_val,
                               le, class_names=class_names_original,
                               slices=None, verbose=False)

    # Save metrics
    if save_metrics:
        plt.savefig(filename + '_pooled.png', bbox_inches="tight")
        plt.close()
        with open(filename, 'w') as f:
            f.write("Training data shape:" + str(X_train.shape))
            f.write('\nnb_classes: ' + str(nb_classes) +
                    '\nslice_length: ' + str(slice_length))
            f.write('\nweights: ' + weights)
            f.write('\nlr: ' + str(lr))
            f.write('\nTest score/loss: ' + str(score[0]))
            f.write('\nTest accuracy: ' + str(score[1]))
            f.write('\nTest results on each slice:\n')
            f.write(str(scores))
            f.write('\n\n Scores when pooling song slices:\n')
            f.write(str(pooling_scores))

    return (scores_dict, pooled_scores_dict)
