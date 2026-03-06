development_flag : bool = True
output_path = "dataset/"

#import modules
import pandas as pd
import os
import kagglehub

def loadDataFrame() -> pd.DataFrame : 
    data = kagglehub.dataset_download(
    "serkantysz/550k-spotify-songs-audio-lyrics-and-genres",
    path="songs.csv")
    return pd.read_csv(data)

def lyricPreprocess (frame : pd.DataFrame )  -> pd.DataFrame:
    # Remove newline chars
    frame['lyrics'] = frame['lyrics'].str.replace(r"\\n", " ", regex=True)
    frame['lyrics'] = frame['lyrics'].str.replace("\n", " ", regex=True)
    frame['lyrics'] = frame['lyrics'].str.replace("\r", " ", regex=True)
    #convert everything to lowercase
    frame['lyrics'] = frame['lyrics'].str.lower()
    #drop duplicates 
    frame['lyrics'] = frame['lyrics'].drop_duplicates()
    #remove short lyrics
    frame : pd.DataFrame = frame[frame['lyrics'].notna() & (frame['lyrics'].str.len() > 250)]     
    
    return frame[['lyrics','genre']]

def getStatisticsOnDataset (frame : pd.DataFrame): 
    #count duplicates based on lyrics 
    counts : pd.Series[int] = frame['lyrics'].value_counts()
    counts : pd.Series[int ]= counts[counts > 1]
    print('Duplicates :' ,frame['lyrics'].duplicated().sum())
    genre_percent : pd.Series[float] = frame['genre'].value_counts(normalize=True) * 100
    print(genre_percent)
    
selectGenres1 : list[str] = ['Rock', 'Pop', 'Electronic', 'Folk']
selectGenres2  : list[str] = [ 'Electronic','Hip-Hop','Jazz','Rock']
selectGenres3  : list[str] = ['Rock', 'Jazz', 'R&B','Hip-Hop']
selectGenres4  : list[str] = ['Rock', 'Jazz', 'Classical', 'Hip-Hop']
selectGenres5  : list[str] = ['Classical', 'Country','Electronic', 'Hip-Hop']

sets : list[list[str]] = [selectGenres1, selectGenres5]

def getBalancedSubset(frame : pd.DataFrame, subset :list[str], dataSetSize : int)  -> pd.DataFrame:
    return frame[frame['genre'].isin(subset)].groupby('genre').sample(n=dataSetSize, random_state=69).reset_index(drop=True)


def saveToDisk(frame : pd.DataFrame, subset : list[str]) :
    prefix : str = "balancedDatasets"
    folderPath : str = f'{prefix}/{"".join(subset)}'
    try:
        os.makedirs(folderPath)
    except FileExistsError : 
        pass
    
    path : str = f'{prefix}/{"".join(subset)}/balancedSubset10k.csv'
    frame.to_csv(path, index=True, encoding='utf-8')
    
    
def preProcessPipeLine(frame : pd.DataFrame, sets : list = sets, dataSetSize : int = 2500): 
    #preprocess the whole dataset
    frame : pd.DataFrame = lyricPreprocess(frame=frame)
    #iterate over both subsets needed for training
    for set in sets:
        subsetFrame : pd.DataFrame = frame
        #get a balanced dataset
        subsetFrame : pd.DataFrame = getBalancedSubset(frame=subsetFrame, subset=set, dataSetSize=dataSetSize)
        #print statistics (optional)
        if development_flag : 
            getStatisticsOnDataset(frame=subsetFrame)
        #save dataset to disk (maybe leave this out?)
        saveToDisk(frame=subsetFrame, subset=set)
     