# 02-GLM-4.1V-Thinking Gradio Deployment

THUDM also provides a gradio interface script to build a ready-to-use Web interface that supports multi-modal inputs such as images, videos, PDFs, and PPTs. Of course, if you call glm4.1v locally, you can modify the corresponding model path.

![image-10.png](images/image-10.png)

```bash
python /root/autodl-tmp/GLM-4.1V-Thinking/inference/trans_infer_gradio.py
```

Users using AutoDL cloud machines can connect according to their own system instructions.

```bash
ssh -F /dev/null -CNg -L 7860:127.0.0.1:7860 [root@connect.nma1.seetacloud.com](mailto:root@connect.nma1.seetacloud.com) -p 36185
```

## Startup Example

![image-11.png](images/image-11.png)

![image-12.png](images/image-12.png)

![image-13.png](images/image-13.png)

![image-14.png](images/image-14.png)