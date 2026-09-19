# 実験指導書の作成

## LaTeX の準備

### 環境構築

[TeX Wiki: Linux > Linux Mint/Debian/Ubuntu](https://texwiki.texjp.org/?Linux%2FLinux%20Mint#texlive) を参照してください。

インストールは次のコマンドで実現できます。

```sh
sudo apt install texlive-lang-japanese texlive-luatex texlive-latex-extra texlive-picture
```

### PDF のコンパイル

次のコマンドで `main.tex` をコンパイルして `out/main.pdf` を生成できます。

```sh
latexmk main.tex
```

なお、次のコマンドで一時ファイルおよび生成ファイルをクリーンできます。

```sh
latexmk -c main.tex
```