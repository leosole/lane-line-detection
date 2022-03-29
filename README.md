# lane-line-detection
Para utilizar esse repositório, modifique o arquivo config.ini com o caminho do vídeo a ser analizado e ajuste os parâmetros

Para rodar:
```
python run.py
```
## Parâmetros
DEBUG: 1 para mostrar o vídeo durante o processo

SAVE_VIDEO: 1 para salvar o vídeo final

FRAME_MEMORY: quantidade de frames utilizados como memória caso não encontre a faixa

FRAME_OVERLAP: quantidade de frames utilizados em conjunto para aumentar as chances de detecção de faixa (só para PAVEMENT = 1)

CONTRAST_FACTOR: fator de contraste utilizado

THRESHOLD: threshold utilizado

PAVEMENT: 1 = asfaltado, 0 = estrada de terra

LANE_HEIGHT: altura padrão da linha, caso apenas uma seja encontrada

## Situação atual
O modo PAVEMENT = 0 ainda está em desenvolvimento
