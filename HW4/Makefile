HW4_0816153.zip: part1/*.cc part2/Makefile part2/main.cc part2/matmul.cc url.txt
	rm -rf HW4_0816153.zip temp
	mkdir temp
	cd temp && mkdir part1 part2
	cp url.txt temp/url.txt
	cp part1/*.cc temp/part1
	cp part2/*.cc temp/part2
	cd temp && zip HW4_0816153.zip -r part1 part2 url.txt
	cd temp && mv HW4_0816153.zip ../HW4_0816153.zip
	rm -rf temp
	printf "\033[92mZip Complete!\033[0m\n"

test: HW4_0816153.zip
	test_hw4

clean:
	rm -rf HW4_0816153.zip temp
