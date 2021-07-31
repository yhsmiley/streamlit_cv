WORKSPACE=/media/data/streamlit_cv

docker run -it \
	--gpus all \
	--net host \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	streamlit
