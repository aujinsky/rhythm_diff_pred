import glob
import os
import csv
import tqdm
import time
if __name__=="__main__":
    import fetch_data
else:
    from utils import fetch_data

songs_dir = 'C:/Users/aujin/AppData/Local/osu!/Songs'
song_list_path = os.path.join('osumania_data', 'mapping_list.csv')
beatmap_data_path = os.path.join('osumania_data', 'beatmap_data.csv')
set_lists = os.listdir(songs_dir)

song_list_f = open(song_list_path, 'w', newline='', encoding='utf-8')
song_list_csv = csv.writer(song_list_f)
beatmap_data_f = open(beatmap_data_path, 'w', newline='', encoding='utf-8')
beatmap_data_csv = csv.writer(beatmap_data_f)
index = 0

for song in tqdm.tqdm(set_lists):
    song_dir = os.path.join(songs_dir, song)
    song_map_dirs = glob.glob(song_dir+"/*.osu")
    for song_map_dir in song_map_dirs:
        with open(song_map_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            bid = -1
            sid = -1
            for line in lines:
                if "BeatmapID" in line: 
                    bid = line.replace("BeatmapID:", "").replace("\n", "")
                elif "BeatmapSetID" in line:
                    sid = line.replace("BeatmapSetID:", "").replace("\n", "")
                if not bid == -1 and not sid == -1:
                    break
        if bid == '-1' and sid == '-1':
            print(bid, sid, song_map_dir)
            continue
        elif bid == '-1' or sid == '-1':
            continue
        else:
            result = fetch_data.api_fetcher(params = {"s": sid, "b": bid})
            if result == None:
                continue
            elif fetch_data.status_check(result):
                index = index + 1
                song_list_csv.writerow([index, sid, bid, song_map_dir])
                beatmap_data_csv.writerow(list(result.values()))
            else:
                continue
        time.sleep(0.038)
        
print("Total " + str(index) + " samples.")