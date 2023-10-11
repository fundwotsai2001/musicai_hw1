import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import dill
import random
import itertools
import numpy as np
from numpy.random import RandomState

import librosa
import librosa.display
import random

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy import stats
from sklearn.metrics import accuracy_score


def visualize_spectrogram(path, duration=None,
                          offset=0, sr=16000, n_mels=128, n_fft=2048,
                          hop_length=512):
    """This function creates a visualization of a spectrogram
    given the path to an audio file."""

    # Make a mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop_length)

    # Convert to log scale (dB)
    log_S = librosa.logamplitude(S, ref_power=1.0)

    # Render output spectrogram in the console
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def create_dataset(artist_folder='artist20', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(f'{save_folder}_train', exist_ok=True)
    os.makedirs(f'{save_folder}_val', exist_ok=True)
    artists = os.listdir(artist_folder)
    # iterate through all artists, albums, songs and find mel spectrogram
    # print(os.listdir(artist_folder))
    # print(artists)
    for artist in artists:
        
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)
            # print(album_songs)
            for song in album_songs:
                if os.path.isdir(os.path.join(album_path,song)):
                    # print(song)
                    song_path = os.path.join(album_path,song,song.split(".")[0],"vocals.wav")
                    # Create mel spectrogram and convert it to the log scale
                    y, sr = librosa.load(song_path, sr=sr)
                    output = random.randint(1, 10)
                    if output <= 9:                        
                        if output == 1:
                            # Time-stretching
                            random_rate = 0.75 + 0.5 * random.random()
                            y = librosa.effects.time_stretch(y, random_rate)
                        if output == 2:
                            # Pitch-shifting
                            random_step = random.randint(0, 5)
                            y = librosa.effects.pitch_shift(y, sr, random_step)
                        
                        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length)
                        if output == 3:
                            # time_masking
                            num_time_bins = S.shape[1]
                            time_mask_length = np.random.randint(low=0, high=num_time_bins // 5)
                            t0 = np.random.randint(low=0, high=num_time_bins - time_mask_length)
                            S[:, t0:t0 + time_mask_length] = 0
                        if output == 4:
                            noise = np.random.randn(S.shape[0], S.shape[1])
                            S = S + 0.0001 * noise
                        if output == 5:
                            num_mel_bins = S.shape[0]
                            freq_mask_length = np.random.randint(low=0, high=num_mel_bins // 5)
                            f0 = np.random.randint(low=0, high=num_mel_bins - freq_mask_length)
                            S[f0:f0 + freq_mask_length, :] = 0
                        log_S = librosa.amplitude_to_db(S, ref=1.0)
                        data = (artist, log_S, song)
                    
                        # Save each song
                        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                        with open(os.path.join(f'{save_folder}_train', save_name), 'wb') as fp:
                            dill.dump(data, fp)
                    else:
                        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length)
                        log_S = librosa.amplitude_to_db(S, ref=1.0)
                        data = (artist, log_S, song)
                    
                        # Save each song
                        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                        with open(os.path.join(f'{save_folder}_val', save_name), 'wb') as fp:
                            dill.dump(data, fp)

def create_testing_dataset(artist_folder='artist20_testing_data', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = os.listdir(artist_folder)
    # iterate through all artists, albums, songs and find mel spectrogram
    # print(os.listdir(artist_folder))
    # print(artists)
    # for artist in artists:
    #     print(artist)
    #     artist_path = os.path.join(artist_folder, artist)
    #     artist_albums = os.listdir(artist_path)

    #     for album in artist_albums:
    #         album_path = os.path.join(artist_path, album)
    #         album_songs = os.listdir(album_path)

    for song in artists:
        # print("notdir")
        if os.path.isdir(os.path.join(os.path.join(artist_folder,song))):
            # print("isdir")
            song_path = os.path.join(artist_folder, song,song.split(".")[0],"vocals.wav")
            # print(song_path)
            # Create mel spectrogram and convert it to the log scale
            y, sr = librosa.load(song_path, sr=sr)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
            print("s",S.shape)
            log_S = librosa.amplitude_to_db(S, ref=1.0)
            print("log_S",log_S.shape)
            data = (log_S)

            # Save each song
            save_name = song
            with open(os.path.join(save_folder, save_name), 'wb') as fp:
                dill.dump(data, fp)
def create_testing_myvoice_dataset(my_file='test_singer.wav', save_folder='my_voice',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    
   
    y, sr = librosa.load(my_file, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                        n_fft=n_fft,
                                        hop_length=hop_length)
    print("s",S.shape)
    log_S = librosa.amplitude_to_db(S, ref=1.0)
    print("log_S",log_S.shape)
    data = (log_S)

    # Save each song
    save_name = my_file.split(".")[-1]
    with open(os.path.join(save_folder, save_name), 'wb') as fp:
        dill.dump(data, fp)
def load_dataset(song_folder_name='song_data',
                 artist_folder='artists',
                 nb_classes=20, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name


def load_test_set(song_folder_name='song_data',
                 artist_folder='artists',
                 nb_classes=20, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)
    # print(song_list)
    # Load the list of artists
    # artist_list = os.listdir(artist_folder)

    # select the appropriate number of classes
    # prng = RandomState(random_state)
    # artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    # artist = []
    spectrogram = []
    song_name = []
    song_length = []
    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
            # print(loaded_song.shape[1])
            # artist.append(loaded_song[0])
            spectrogram.append(loaded_song)
            song_length.append(loaded_song.shape[1])
            # print("testing",loaded_song.shape)
            song_name.append(int(song.split(".")[0]))
    # print("song_length",len(song_length))
    # print("spectrogram",len(spectrogram[0]))
    return spectrogram,song_length,song_name

def load_dataset_album_split(song_folder_name='song_split_data',
                             artist_folder='artists',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    train_albums = []
    # test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        albums = os.listdir(os.path.join(artist_folder, artist))
        random.shuffle(albums)
        # test_albums.append(artist + '_%%-%%_' + albums.pop(0))
        val_albums.append(artist + '_%%-%%_' + albums.pop(0))
        train_albums.extend([artist + '_%%-%%_' + album for album in albums])
    # print("train_albums",train_albums)
    # print("val_albums",val_albums)
    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train,Y_val = [], []
    X_train,X_val = [], []
    S_train,S_val = [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        artist, album, song_name = song.split('_%%-%%_')
        artist_album = artist + '_%%-%%_' + album

        if loaded_song[0] in artists:
            if artist_album in train_albums:
                Y_train.append(loaded_song[0])
                X_train.append(loaded_song[1])
                S_train.append(loaded_song[2])
            elif artist_album in val_albums:
                Y_val.append(loaded_song[0])
                X_val.append(loaded_song[1])
                S_val.append(loaded_song[2])
    # print(len(Y_train), len(X_train), len(S_train), \
    #        len(Y_val), len(X_val), len(S_val))
    return Y_train, X_train, S_train, \
           Y_val, X_val, S_val


def load_dataset_song_split(song_folder_name='song_data',
                            artist_folder='artists',
                            nb_classes=20,
                            test_split_size=0,
                            validation_split_size=0.1,
                            random_state=42):
    Y, X, S = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    # X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
    #     X, Y, S, test_size=test_split_size, stratify=Y,
    #     random_state=random_state)

    # Create a validation to be used to track progress
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(
        X, Y, S, test_size=validation_split_size,
        shuffle=True, stratify=Y, random_state=random_state)

    return Y_train, X_train, S_train, \
           Y_val, X_val, S_val


def slice_val_songs(X, Y, S, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []
    count = {}
    # Slice up songs using the length specified
    for i, song in enumerate(X):
        # print(song.shape)
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(Y[i])
            song_name.append(S[i])
    # print("singer count:",count)
    
    return np.array(spectrogram), np.array(artist), np.array(song_name)
def slice_train_songs(X, Y, S, length=911, augmentation_multiplier=0.25):
    """Slices the spectrogram into sub-spectrograms according to length
    and performs data augmentation by concatenating slices from different songs."""

    # Map to store slices of songs by each artist
    artist_slices = {}
    # length = int(length/2)
    # # Slice up songs using the length specified
    # for i, song in enumerate(X):
    #     slices = int(song.shape[1] / length)
    #     artist = Y[i]
    #     for j in range(slices):
    #         slice_ = song[:, length * j:length * (j + 1)]
    #         artist = Y[i]
            
    #         # Add the slices to the artist_slices map
    #         if artist in artist_slices:
    #             artist_slices[artist].append(slice_)
    #         else:
    #             artist_slices[artist] = [slice_]
    
    augmented_song_names = []
    augmented_slices = []
    augmented_artists = []    
    # for artist, slices in artist_slices.items():
    #     num_slices = len(slices)
    #     for i in range(int(len(slices))):
    #         if slices:
    #         # Select 4 random slices from the same artist and concatenate them
    #             if len(slices) < 2:
    #                 # print(len(slices))
    #                 break
    #             slice1 = slices.pop(np.random.randint(len(slices)))
    #             slice2 = slices.pop(np.random.randint(len(slices)))
    #             augmented_slice = np.concatenate((slice1, slice2), axis=1)
    #             augmented_slices.append(augmented_slice)
    #             augmented_artists.append(artist)
    # length *= 2
    for i, song in enumerate(X):
        # print(song.shape)
        slices = int(song.shape[1] / length)
        for j in range(slices):
            single_slice = song[:, length * j:length * (j + 1)]
            # Ensure that single_slice has the same shape as augmented_slice
            # This might involve padding, cropping, or other modifications to single_slice
            augmented_slices.append(np.array(single_slice))
            augmented_artists.append(Y[i])
    # Combine the two lists into pairs using zip
    combined_lists = list(zip(augmented_slices, augmented_artists))

    # Shuffle the combined list
    random.shuffle(combined_lists)

    # Unzip the shuffled pairs back into separate lists
    augmented_slices, augmented_artists = zip(*combined_lists)
    return np.array(augmented_slices), np.array(augmented_artists), np.array(augmented_song_names)
def slice_test_songs(X, length=94):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    # artist = []
    spectrogram = []
    song_slices = []
    # song_name = []
    # Slice up songs using the length specified
    for i, song in enumerate(X):
        # print(song.shape)
        slices = int(song.shape[1] / length)
        song_slices.append(slices)
        # print(f'{slices} = int({song.shape[1]} / {length})')
        # print(range(slices))
        for j in range(slices):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            # print(j)
            # artist.append(Y[i])
            # song_name.append(S[i])
    # print("sliced spectralgram shape",np.array(spectrogram).shape)
    return np.array(spectrogram),np.array(song_slices)


def create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                             n_fft=2048, hop_length=512):
    """Create a spectrogram from a randomly selected song
     for each artist and plot"""

    # get list of all artists
    artists = os.listdir(artist_folder)

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 12), sharex=True,
                           sharey=True)

    row = 0
    col = 0

    # iterate through artists, randomly select an album,
    # randomly select a song, and plot a spectrogram on a grid
    for artist in artists:
        # print(artist)
        # Randomly select album and song
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        album = random.choice(artist_albums)
        album_path = os.path.join(artist_path, album)
        album_songs = os.listdir(album_path)
        song = random.choice(album_songs)
        song_path = os.path.join(album_path, song)

        # Create mel spectrogram
        y, sr = librosa.load(song_path, sr=sr, offset=60, duration=3)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)

        # Plot on grid
        plt.axes(ax[row, col])
        librosa.display.specshow(log_S, sr=sr)
        plt.title(artist)
        col += 1
        if col == 5:
            row += 1
            col = 0

    fig.tight_layout()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    




def plot_history(history, title="model accuracy"):
    """
    This function plots the training and validation accuracy
     per epoch of training
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    return


def predict_artist(model, X, Y, S,
                   le, class_names,
                   slices=None, verbose=False,
                   ml_mode=False):
    """
    This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    print("Test results when pooling slices by song and voting:")
    # Obtain the list of songs
    songs = np.unique(S)

    prediction_list = []
    actual_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        X_song = X[S == song]
        Y_song = Y[S == song]

        # If not using full song, shuffle and take up to a number of slices
        if slices and slices <= X_song.shape[0]:
            X_song, Y_song = shuffle(X_song, Y_song)
            X_song = X_song[:slices]
            Y_song = Y_song[:slices]

        # Get probabilities of each class
        predictions = model.predict(X_song, verbose=0)

        if not ml_mode:
            # Get list of highest probability classes and their probability
            class_prediction = np.argmax(predictions, axis=1)
            class_probability = np.max(predictions, axis=1)

            # keep only predictions confident about;
            prediction_summary_trim = class_prediction[class_probability > 0.5]

            # deal with edge case where there is no confident class
            if len(prediction_summary_trim) == 0:
                prediction_summary_trim = class_prediction
        else:
            prediction_summary_trim = predictions

        # get most frequent class
        prediction = stats.mode(prediction_summary_trim)[0][0]
        actual = stats.mode(np.argmax(Y_song))[0][0]

        # Keeping track of overall song classification accuracy
        prediction_list.append(prediction)
        actual_list.append(actual)

        # Print out prediction
        if verbose:
            print(song)
            print("Predicted:", le.inverse_transform(prediction), "\nActual:",
                  le.inverse_transform(actual))
            print('\n')

    # Print overall song accuracy
    actual_array = np.array(actual_list)
    prediction_array = np.array(prediction_list)
    cm = confusion_matrix(actual_array, prediction_array)
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Confusion matrix for pooled results' +
                                ' with normalization')
    class_report = classification_report(actual_array, prediction_array,
                                         target_names=class_names)
    print(class_report)
    accuracy = accuracy_score(actual_array, prediction_array)
    print("Accuracy:", accuracy)
    class_report_dict = classification_report(actual_array, prediction_array,
                                              target_names=class_names,
                                              output_dict=True)
    return (class_report, class_report_dict)


def encode_labels(Y, le=None, enc=None):
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y).reshape(N, 1)
    else:
        Y_le = le.transform(Y).reshape(N, 1)

    # convert into one hot encoding
    if enc is None:
        enc = preprocessing.OneHotEncoder()
        Y_enc = enc.fit_transform(Y_le).toarray()
    else:
        Y_enc = enc.transform(Y_le).toarray()
    # print(le.classes_)
    # return encoders to re-use on other data
    return Y_enc, le, enc


def simple_encoding(Y, le=None):
    """Encodes target variables into numbers"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y)
    else:
        Y_le = le.transform(Y)

    # return encoders to re-use on other data
    return Y_le, le


if __name__ == '__main__':

    # configuration options
    create_data = True
    create_visuals = False
    save_visuals = False

    if create_data:
        # create_dataset(artist_folder='artist20', save_folder='song_split_data',
        #                sr=16000, n_mels=128, n_fft=2048,
        #                hop_length=512)
        # create_testing_dataset(artist_folder='artist20_testing_data', save_folder='song_split_test_data',
        #            sr=16000, n_mels=128,
        #            n_fft=2048, hop_length=512)
        create_testing_myvoice_dataset(my_file='test2.wav', save_folder='my_voice',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512)

    if create_visuals:
        # Create spectrogram for a specific song
        visualize_spectrogram(
            'artists/u2/The_Joshua_Tree/' +
            '02-I_Still_Haven_t_Found_What_I_m_Looking_For.mp3',
            offset=60, duration=29.12)

        # Create spectrogram subplots
        create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                                 n_fft=2048, hop_length=512)
        if save_visuals:
            plt.savefig(os.path.join('spectrograms.png'),
                        bbox_inches="tight")
