for i in image*; do
    # 'image'를 제외한 숫자 부분 추출
    num=$(echo "$i" | sed 's/image//')
    # 숫자를 9자리로 맞추어 새로운 이름 생성
    new_name=$(printf "image%09d" "$num")
    mv "$i" "$new_name"
done
