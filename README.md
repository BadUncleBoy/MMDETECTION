# MMDETECTION

### Segmentation Arciitecture
<img src="https://raw.githubusercontent.com/BadUncleBoy/MMDETECTION/gggs/data/jpgs/seg_architecture.png" width="1024">


|   %        |             Method        | Backbone     | AP     | AP50     | AP75 | APS     | APM  | APL  | AR100 | ARS  | ARM     | ARL  |
| :--------: | :-----------------------: | :----------: | :----: | -------- | :--: | :-----: | :--: | :--: | :---: | :--: | :-----: | :--: |
| **VG1000** |         **HKRM**          |  ResNet101   |  7.8   | 13.4     | 8.1  |   4.1   | 8.1  | 12.7 | 22.7  | 9.6  |  20.8   | 31.4 |
| **VG1000** | **Reasoning-RCNNR w FPN** |  ResNet101   |  8.2   | 13.3     | 8.5  |   3.4   | 8.3  | 14.0 | 23.5  | 8.8  |  21.7   | 32.9 |
| **VG1000** |      **SGRN w FPN**       |  ResNet101   |  8.1   | 13.6     | 8.4  |   4.4   | 8.2  | 12.8 | 26.2  | 12.4 |  23.9   | 34.0 |
| **VG1000** |  **GGRCNN with FPN(u)**   |   ResNet50   |  6.7   | 11.7     | 6.8  |   4.0   | 7.1  | 9.6  | 25.2  | 15.4 |  24.0   | 28.3 |
| **VG1000** |  **GGRCNN with FPN(r)**   |   ResNet50   |  7.7   | 13.5     | 7.7  |   4.9   | 8.4  | 11.4 | 28.5  | 17.3 |  27.8   | 32.1 |







|   %    |       Method        | Backbone  | APall | APrare | APless | APmiddle | APmany | ARall | ARrare | ARless | ARmiddle | ARmany |
| :----: | :-----------------: | --------- | :---: | :----: | :----: | :------: | :----: | :---: | :----: | :----: | :------: | :----: |
| VG1000 | GGRCNN with FPN (n) | ResNet50  |  6.7  |  3.7   |  7.6   |   14.4   |  15.0  | 25.2  |  15.7  |  28.5  |   41.3   |  44.7  |
| VG1000 | GGRCNN with FPN (r) | ResNet50  |  7.7  |  5.4   |  8.1   |   14.8   |  15.0  | 28.5  |  21.2  |  30.5  |   41.9   |  44.7  |
| VG1000 | GGRCNN with FPN (x) | ResNet50  |  7.7  |        |        |          |        |       |        |        |          |        |
| VG1000 | FasterRCNN with FPN | ResNet50  |  7.6  |  5.2   |  8.0   |   15.0   |  15.2  |       |  20.6  |  30.1  |   42.1   |  44.7  |
| VG1000 |   GSRCNN with FPN   | ResNet 50 |  7.9  |  6.1   |  8.2   |   14.5   |  14.8  | 31.7  |  28.7  |  31.4  |   39.0   |  41.5  |

