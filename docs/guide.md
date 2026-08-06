# CÁC HƯỚNG DẪN VỀ DỰ ÁN 
## 1. Một số lệnh conda: 
### Cài đặt conda 
- Cài đặt conda: https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install

- Với mỗi loại Terminal: Bash, zsh,.. muốn dùng conda thì phải init cho loại terminal đó 

`conda init zsh`hoặc `conda init bash`

- Sau khi `init` thì chạy các lệnh để thực thi các file lệnh của từng loại Terminal 

```
source ~/.bashrc
zsh ~/.zsh
exec fish 
``` 
### Lệnh 
`conda create --prefix ./env python=3.11`

`conda activate ./venv`

`python -m pip install -r requirements.txt`
