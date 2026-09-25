metadata
license: cc-by-nc-4.0
task_categories:
  - object-detection
language:
  - en
pretty_name: CrowdHuman
size_categories:
  - 10K<n<100K
CrowdHuman: A Benchmark for Detecting Human in a Crowd
🏠 homepage: https://www.crowdhuman.org/
📄 paper: https://arxiv.org/pdf/1805.00123
CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

image/png
Volume, density and diversity of different human detection datasets. For fair comparison, we only show the statistics of training subset.

🔍 Samples
image/png
image/png
image/png
image/png
image/png
image/png
image/png
image/png
image/png
📁 Files
CrowdHuman_train01.zip
CrowdHuman_train02.zip
CrowdHuman_train03.zip
CrowdHuman_val.zip
CrowdHuman_test.zip
annotation_train.odgt
annotation_val.odgt
🖨 Data Format
We support annotation_train.odgt and annotation_val.odgt which contains the annotations of our dataset.

What is odgt?
odgt is a file format that each line of it is a JSON, this JSON contains the whole annotations for the relative image. We prefer using this format since it is reader-friendly.

Annotation format
JSON{
    "ID" : image_filename,
    "gtboxes" : [gtbox], 
}

gtbox{
    "tag" : "person" or "mask", 
    "vbox": [x, y, w, h],
    "fbox": [x, y, w, h],
    "hbox": [x, y, w, h],
    "extra" : extra, 
    "head_attr" : head_attr, 
}

extra{
    "ignore": 0 or 1,
    "box_id": int,
    "occ": int,
}

head_attr{
    "ignore": 0 or 1,
    "unsure": int,
    "occ": int,
}
Keys in extra and head_attr are optional, it means some of them may not exist
extra/head_attr contains attributes for person/head
tag is mask means that this box is crowd/reflection/something like person/... and need to be ignore(the ignore in extra is 1)
vbox, fbox, hbox means visible box, full box, head box respectively
⚠️ Terms of use:
by downloading the image data you agree to the following terms:

You will use the data only for non-commercial research and educational purposes.
You will NOT distribute the above images.
Megvii Technology makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
You accept full responsibility for your use of the data and shall defend and indemnify Megvii Technology, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.
🏆 Related Challenge
Detection In the Wild Challenge Workshop 2019
📚 Citaiton
Please cite the following paper if you use our dataset.

@article{shao2018crowdhuman,
  title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
  author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:1805.00123},
  year={2018}
}
👥 People
Shuai Shao*
Zijian Zhao*
Boxun Li
Tete Xiao
Gang Yu
Xiangyu Zhang
Jian Sun