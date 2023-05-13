# ライブラリのインポート
import streamlit as st
from model import predict#, gradcam
from PIL import Image


# タイトル
st.title('イラスト判定 :smirk:')

# 選択
option = st.sidebar.selectbox(
    '描いたイラストは何ですか？',
    ('イヌ', 'ネコ', 'オオカミ', 
     'ゾウ', 'ウマ', 'パンダ', 
     'シカ', 'ウシ', 'キツネ', 
     'タヌキ', 'トラ', 'イノシシ')
)

# ファイルの読み込み
img = st.sidebar.file_uploader(
    label = "描いたイラストを読み込み",
    type=['jpeg', 'jpg', 'png'],
    accept_multiple_files = False
)

# 推論
if img != None:
	pred = predict(Image.open(img).convert('RGB'))
else:
	pred = '???'

col1, col2 = st.columns(2)
# 読み込んだ画像の表示
with col1:
	if img != None:
		st.image(
			img,
			caption = 'あなたのイラスト',
			width = 256
		)
	else:
		st.image(
			'./dummy.png',
			caption = '画像を読み込んでね！',
			width = 256
		)

# 推論結果の表示
list = ['イノシシ', 'ネコ', '牛', 'シカ', 'イヌ', 'ゾウ', 'キツネ', 'ウマ', 'パンダ', 'タヌキ', 'トラ', 'オオカミ']
with col2:
	st.write('\n')
	st.write('\n')
	st.write('\n')
	st.write('\n')
	st.write('\n')
	st.write('\n')
	st.write('\n')
	st.write('\n')
	if img != None:
		st.write(format(pred.cpu().detach().numpy().copy()[0][list.index(option)] * 100, '.2f'), '% で', option, 'です')
	else:
		st.write('あなたのイラストは')
		st.write('???')
		st.write('と判定されました')

# Grad-CAMによる可視化
# if img != None:
# 	gradimg = gradcam(Image.open(img).convert('RGB'))

# 	st.image(
# 		gradimg,
# 		caption = 'AIはここに注目したよ',
# 		width = 256
# 	)