#!/bin/bash

# mkcd folder := mkdir folder && cd folder
mkcd() { mkdir -p "$1" && cd "$1"; }

mkcd datasets

# Download Subjects from Dreambooth
git clone https://github.com/google/dreambooth.git
cd dreambooth
mv dataset ../subjects 
cd ..
rm -rf dreambooth

# Download Styles 
mkcd styles
mkcd 3d_rendering && curl -L -o hatiful-yosa-0e6nHU8GRUY-unsplash.jpg "https://unsplash.com/photos/0e6nHU8GRUY/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA0ODAxfA" && cd ../
mkcd 3d_rendering2 && curl -L -o shubham-dhage-t0Bv0OBQuTg-unsplash.jpg "https://unsplash.com/photos/t0Bv0OBQuTg/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzAxOTYwfA" && cd ../
mkcd 3d_rendering3 && wget "https://img.freepik.com/free-psd/three-dimensional-real-estate-icon-mock-up_23-2149729145.jpg" && cd ../
mkcd 3d_rendering4 && curl -L -o vadim-bogulov-rdHrrFA1KKg-unsplash.jpg "https://unsplash.com/photos/rdHrrFA1KKg/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MXx8fGVufDB8fHx8MTc1ODcxMzk2OHw" && cd ../
mkcd melting_golden_rendering && curl -L -o simon-lee-Prx96KdmWj0-unsplash.jpg "https://unsplash.com/photos/Prx96KdmWj0/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA1NTQxfA" && cd ../
mkcd glowing_3d_rendering && curl -L -o peter-bo-1RUHvfnaCWY-unsplash.jpg "https://unsplash.com/photos/1RUHvfnaCWY/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MXx8fGVufDB8fHx8MTc1ODcxNDA2OXw" && cd ../
mkcd glowing && curl -L -o mathew-schwartz-RWrbY8j9GPo-unsplash.jpg "https://unsplash.com/photos/RWrbY8j9GPo/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MXx8fGVufDB8fHx8MTc1ODcxMzgxMHw" && cd ../
mkcd sticker && wget "https://img.freepik.com/vettori-gratuito/adesivo-albero-di-pino-su-sfondo-bianco_1308-75956.jpg" && cd ../
mkcd watercolor_painting && curl -L -o markus-spiske-6dY9cFY-qTo-unsplash.jpg "https://unsplash.com/photos/6dY9cFY-qTo/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA0OTIyfA" && cd ../
mkcd watercolor_painting2 && curl -L -o david-clode-H9g_HE6ZgGA-unsplash.jpg "https://unsplash.com/photos/H9g_HE6ZgGA/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA2MDA2fA" && cd ../
mkcd watercolor_painting3 && curl -L -o fuu-j-jI3Lp0FYEz0-unsplash.jpg "https://unsplash.com/photos/jI3Lp0FYEz0/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA2MDI0fA" && cd ../
mkcd watercolor_painting4 && curl -L -o mcgill-library-kHuCUkkExbc-unsplash.jpg "https://unsplash.com/photos/kHuCUkkExbc/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA2MDM5fA" && cd ../
mkcd watercolor_painting5 &&  curl -L -o birmingham-museums-trust-0pJPixfGfVo-unsplash.jpg "https://unsplash.com/photos/0pJPixfGfVo/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA0MjIwfA" && cd ../
mkcd watercolor_painting6 && curl -L -o fuu-j-6L4jcwgDNNE-unsplash.jpg "https://unsplash.com/photos/6L4jcwgDNNE/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MXx8fGVufDB8fHx8MTc1ODcxNDIxN3w" && cd ../
mkcd watercolor_painting7 && curl -L -o birmingham-museums-trust-X2QwsspYk_0-unsplash.jpg "https://unsplash.com/photos/X2QwsspYk_0/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MXx8fGVufDB8fHx8MTc1ODcxNDMzOHw" && cd ../
mkcd flat_cartoon_illustration && wget "https://img.freepik.com/free-vector/biophilic-design-workspace-abstract-concept_335657-3081.jpg" && cd ../
mkcd flat_cartoon_illustration2 && wget "https://img.freepik.com/free-vector/young-woman-walking-dog-leash-girl-leading-pet-park-flat-illustration_74855-11306.jpg" && cd ../
mkcd line_drawing && wget "https://upload.wikimedia.org/wikipedia/commons/d/de/Van_Gogh_Starry_Night_Drawing.jpg" && cd ../
mkcd kid_crayon_drawing && wget "https://raw.githubusercontent.com/styledrop/styledrop.github.io/refs/heads/main/images/assets/image_6487327_crayon_02.jpg" && cd ../
mkcd oil_painting && wget "https://upload.wikimedia.org/wikipedia/commons/6/66/VanGogh-starry_night_ballance1.jpg" && cd ../
mkcd oil_painting2 && wget "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/1024px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg" && cd ../
mkcd oil_painting3 && wget "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg/1024px-Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg" && cd ../
mkcd abstract_rainbow_colored_flowing_smoke_wave_design && wget "https://img.freepik.com/free-psd/abstract-background-design_1297-124.jpg" && cd ../
mkcd wooden_sculpture && curl -L -o paolo-chiabrando-CuWq_99U0xs-unsplash.jpg "https://unsplash.com/photos/CuWq_99U0xs/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzA1NzIzfA" && cd ../
mkcd black_statue && curl -L -o donovan-reeves-gZzUo--BTZ4-unsplash.jpg "https://unsplash.com/photos/gZzUo--BTZ4/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzU4NzAyNTkyfA" && cd ../

# this style need to be manually downloaded
# mkdir cartoon_line_drawing # download the jpeg from here "https://www.instagram.com/p/CqwU1bavm0T/"

cd ../
mkdir test_datasets
mkdir test_datasets/subjects

cp -r subjects/can test_datasets/subjects
cp -r subjects/cat2 test_datasets/subjects
cp -r subjects/dog8 test_datasets/subjects
cp -r subjects/teapot test_datasets/subjects
cp -r subjects/wolf_plushie test_datasets/subjects

mkdir test_datasets/styles
cp -r styles/3d_rendering4 test_datasets/styles
cp -r styles/flat_cartoon_illustration test_datasets/styles
cp -r styles/glowing test_datasets/styles
cp -r styles/oil_painting2 test_datasets/styles
cp -r styles/watercolor_painting3 test_datasets/styles


mkdir val_datasets
mkdir val_datasets/subjects

cp -r subjects/bear_plushie val_datasets/subjects
cp -r subjects/clock val_datasets/subjects
cp -r subjects/dog2 val_datasets/subjects
cp -r subjects/dog8 val_datasets/subjects


mkdir val_datasets/styles
cp -r styles/3d_rendering2 val_datasets/styles
cp -r styles/oil_painting val_datasets/styles
cp -r styles/watercolor_painting val_datasets/styles

