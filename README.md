# AdvCam-pytorch-impl

forked from [https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles).

This is a Pytorch version implementation of AdvCam, which runs correctly on Windows 11. Please refer to the original repo for various parameter settings.

To run this code, you only need to run`advcam_main_torch.py`, 

```shell
python advcam_main_torch.py
```

The basic code framework is implemented by New Bing, and I have made modifications based on it.

It should be noted that during the testing of the code, I removed the original Style Loss section and rewritten it, without using content and style masks in the new version.

In addition, I have also removed the physical adaptation section from the original version. If necessary, you can add it yourself.

fork自 [https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles).

这是AdvCam的Pytorch版本实现，它在Windows11上正确运行。有关各种参数设置，请参阅原仓库。

要运行代码，只需要运行`advcam_main_torch.py`即可。

基本的代码框架是由New Bing实现的，我在其基础上进行了修改。

需要注意的是，在测试代码的过程中，我删除了原来的Style_Loss部分并对其进行了重写，在新版本中没有使用content mask和style mask。

此外，我还从原始版本中删除了物理适应部分。如有必要，您可以自己添加。
