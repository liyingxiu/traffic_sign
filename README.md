1. 生成pb文件， 使用converter.py, 修改其中的保存文件名

2. 转换为xml和bin文件， 使用openvino的model_optimizer
 
```bash
activate "your conda env"
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
cd "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer"
python mo.py --input_model xxx.pb --data-type

3. 运行模型
```bash
python classification_sample.py -m xxx.xml -i xxx.jpg
```

参考资料
- https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html
- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
- https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html