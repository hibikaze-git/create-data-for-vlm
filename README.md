# create-data-for-vlm
- vlmの学習用データの合成・翻訳を行うリポジトリ
- 各データの準備方法は./datasets以下のディレクトリに記載

# 使い方
## インストール
```
git clone https://github.com/hibikaze-git/create-data-for-vlm.git
```

## 環境構築
### docker
- docker-compose.ymlのenvironmentとdevice_idsを環境に合わせて変更し、以下を実行
```
docker compose --profile ubuntu-cuda build
docker compose --profile ubuntu-cuda up -d
docker compose exec ubuntu-cuda bash
```

### docker以外
- ./docker/Dockerfileを参照して環境構築
