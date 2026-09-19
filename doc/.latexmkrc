# .latexmkrc

# 1. 使用するコンパイラを LuaLaTeX に指定
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode %O %S';
$pdf_mode = 4; # PDF生成に LuaLaTeX を使用する設定

# 2. ビューア（プレビューア）の起動設定 (お好みで。1で自動起動)
$preview_behavior = 1;

# 3. 必要に応じて、中間ファイルをクリーンアップする際の対象を指定
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml bbl blg lof lot out toc';

# Ex. 一時ファイルの保存先
$aux_dir = 'build';
$out_dir = 'out';