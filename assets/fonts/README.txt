NotoSansJP-Bold.ttf: static wght=700 instance of Noto Sans JP (variable font),
made with: fonttools varLib.instancer "NotoSansJP[wght].ttf" wght=700 --update-name-table
Noto Sans JP is (c) Google LLC, licensed under the SIL Open Font License 1.1:
https://openfontlicense.org  (source: https://github.com/notofonts/noto-cjk)

render.py passes this folder to libass as `fontsdir`, so captions do not depend on
a system-wide install. libass renders variable fonts in their default (Regular)
weight, which is why a static Bold file is kept here.
