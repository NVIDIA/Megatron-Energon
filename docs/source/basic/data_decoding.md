<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Data Decoding

When iterating your dataset, the first processing step that each sample goes through, even before [](sample-loading), is the decode step.

As explained [here](wds-format), each sample in the underlying `.tar` file can have multiple entries with different extensions.
Just like WebDataset itself, energon uses the [auto decode](https://rom1504.github.io/webdataset/api/webdataset/autodecode.html) functionality of WebDataset to transform those raw bytes from the tar file entries
to usable objects. Images for example shall be decoded to arrays or tensors of pixel data.

In Energon, the class responsible for data decoding is {py:class}`DefaultDecoderWebdatasetFactory <megatron.energon.DefaultDecoderWebdatasetFactory>`. In its `__init__` method, it contains a piece of code that initializes the auto decoder and tells it which data types shall be decoded and how.

Decoders typically convert

* Text to strings (instead of bytes)
* JSON to dictionaries and lists
* Images, videos and audios to pixel tensors or audio sample tensors
    * For other options see below
* Torch PTH files to tensors
* NPY files to numpy arrays


## How to control data decoding

When calling {py:func}`get_train_dataset <megatron.energon.get_train_dataset>` or {py:func}`get_val_dataset <megatron.energon.get_val_dataset>`, you can pass one or more of the following arguments, which will be passed on to {py:class}`DefaultDecoderWebdatasetFactory <megatron.energon.DefaultDecoderWebdatasetFactory>`:

* `auto_decode` (bool)
    * Set to `False` to disable all automatic decoding of the data. In your sample loader, you will then get raw bytes. The default setting is `True`
* `image_decode` (str)
    * Can be set to an image decoder from webdataset. Here are some examples:
        * `pil`: Returns the image as a PIL image
        * `torchrgb` Returns the image as a torch tensor with 3 color channels.
    * For more options, check out the [official documentation](https://rom1504.github.io/webdataset/api/webdataset/autodecode.html#ImageHandler).
* `av_decode` (str)
    * Can be one of `AVDecoder`, `torch`, `pyav`. The default is `AVDecoder` which is explained [below](av-decoder).
    * The option `torch` would decode video and audio entirely and return them as tensors.
    * The `pyav` option is for advanced use cases where you need direct access to the object returned by `av.open()`
* `video_decode_audio` (bool)
    * If `True`, videos that have an audio track will decode both the video and the audio. Otherwise, only the video frames are decoded.


(av-decoder)=
## AV Decoder (Audio and Video)

Energon comes with code to efficiently decode compressed video and audio files such as MP4, MP3 or WAV.
It integrates a library called `fastseek` that allows to quickly extract sub-clips from longer videos.
The actual decoding is done by PyAV which in turn relies on ffmpeg.

When choosing `AVDecoder` as the option for `av_decode` above, an object of the type {py:class}`AVDecoder <megatron.energon.av.AVDecoder>` (click to see methods) will be returned. 
At this point, the file has not yet been decoded, but in your [custom sample loader](custom-sample-loader) or in your [TaskEncoder](../basic/task_encoder), you can read parts or all of the file using the methods of `AVDecoder`.

For example to extract from a video, the first 2 seconds and the 2 seconds starting after 4 seconds:
```py
dat = av_decoder.get_clips(
    video_clip_ranges=[(0, 2), (4, 6)],
    video_unit="seconds",
)

# Print both clips' tensor shapes
print(dat.video_clips[0].shape)
print(dat.video_clips[1].shape)
```
The return value of {py:meth}`get_clips <megatron.energon.av.AVDecoder.get_clips>` will be an object of type {py:class}`AVData <megatron.energon.av.AVData>`. It may contain video or audio tensors or both, depending on what you requested. It will also contain the timestamps of the actually returned clips.

Click on {py:class}`AVData <megatron.energon.av.AVData>` to see the fields and the shapes of the tensors.


You can also use helper methods to extract multiple clips at equal distances all at once:
```py
from megatron.energon.av import get_clips_uniform

dat = get_clips_uniform(
    av_decoder=av_decoder, num_clips=5, clip_duration_seconds=3, request_audio=True
)
```
This would extract 5 clips, each 3 seconds long also including the corresponding audio clips.
The first clip will start at position 0 and the last clip would end at the end of the video.
This is essentially just a convenience wrapper around `get_clips`, so the code may be a good starting point if you're looking to write a custom extraction logic.

We also provide a method {py:func}`get_single_frames_uniform <megatron.energon.av.get_single_frames_uniform>` which will return a tensor of frames directly instead of an `AVData` object.


The simplest case is to decode the whole video or audio or both:
```py
dat = av_decoder.get_video()

# or
dat = av_decoder.get_audio()

# or
dat = av_decoder.get_video_with_audio()
```