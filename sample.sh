SRC=/mnt/cpgarchive/archives/toxicology/open-tg-gates/images
DEST=./dataset/wsi

mkdir -p "$DEST"

for d in "$SRC"/*/; do
  disease=$(basename "$d")
  kidney_dir="$d/kidney"

  if [ -d "$kidney_dir" ]; then
    mkdir -p "$DEST/$disease"
    find "$kidney_dir" -maxdepth 1 -type f -iname "*.svs" | sort | head -n 1 | while read -r f; do
      rsync -avh --progress "$f" "$DEST/$disease/"
    done
  else
    echo "No kidney folder for $disease, skipping"
  fi
done
