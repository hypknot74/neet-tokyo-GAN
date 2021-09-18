from googleapiclient.discovery import build
from oauth2client.tools import argparser


DEVELOPER_KEY = <自分のGCP Developer key>
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
CUSTOM_NAME = "neet-tokyo" #チャンネルのcustom url 例：https://www.youtube.com/c/<ここ>

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
  developerKey=DEVELOPER_KEY)

search_response = youtube.search().list(
  q=CUSTOM_NAME,
  part="snippet",
  type='channel',
  maxResults=1,
  order="relevance", #関連順にソート
  pageToken=''
  ).execute()

print(search_response.get("items", [])[0])


DEVELOPER_KEY = <自分のGCP Developer key>
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
ChannelId = "UC4kUV8uGKmoQUs1HLzf7jsg" #上で出力したチャンネルID
videos = [] #動画のURLを格納
thums = [] #サムネイルのURLを格納

def youtube_search(pagetoken):
  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  search_response = youtube.search().list(
    q=CUSTOM_NAME,
    part="snippet",
    channelId= ChannelId,
    maxResults=50,
    order="viewCount", #日付順にソート
    pageToken=pagetoken
    ).execute()


  for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
      videos.append(search_result["id"]["videoId"])
      thums.append(search_result["snippet"]["thumbnails"]["high"]["url"])

  try:
      nextPagetoken =  search_response["nextPageToken"] #nextPageTokenが返ってくる限り処理を繰り返す
      youtube_search(nextPagetoken)
  except:
      return

youtube_search('')


print(len(videos))
print(len(thums))


videos[:3]


thums[:3]


pip install youtube_dl


from __future__ import unicode_literals
import youtube_dl

import cv2
import os 
import glob
from tqdm import tqdm


import wandb


wandb.init(project="neet tokyo")

OUTPUT_PATH = './dataset'
N_IMGS = 20


for video_id in tqdm(videos[14+89+106:250]):
  # 動画ダウンロード
  ydl_opts = {}
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

  video_paths1 = glob.glob('*.mp4')
  video_paths2 = glob.glob('*.mkv')
  video_paths = video_paths1 + video_paths2

  if len(video_paths) == 1:
    video_path = video_paths[0]
  else :
    for v in video_paths:
      os.remove(v)
    continue

  root, ext = os.path.splitext(video_path)
  basename = os.path.basename(root)
  output_dir = os.path.join(OUTPUT_PATH, basename)

  os.makedirs(output_dir, exist_ok=True)

  # 動画のカット
  cap = cv2.VideoCapture(video_path)

  cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
  fps = cap.get(cv2.CAP_PROP_FPS)

  video_len = frame_count / fps

  # 抽出したい開始or終了時間
  begin = 5
  end = video_len - 22

  # 切り出すフレーム
  cutout_interval = (end - begin) * fps / N_IMGS
  cutout_frames = [int(begin*fps + i*cutout_interval) for i in range(N_IMGS)]

  for frame_num in cutout_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(output_dir, f'sample_{frame_num}.jpg'), frame)

  os.remove(video_path)



