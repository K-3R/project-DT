# !/bin/bash

## git initialization
git init

## git setup
git add .
git branch -m master main
git remote add origin http://ssl-git/team-sr/2dgs.git

## merge 해서 push 하기
git fetch origin
git merge origin/main --allow-unrelated-histories -m "merge gitlab readme"
git checkout --ours README.md
git add README.md
git commit --no-edit

## 끝!
git push origin main
git pull origin main