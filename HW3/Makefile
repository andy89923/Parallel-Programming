HW3_0816153.zip: part1/cg_impl.c part2/breadth_first_search/bfs.cpp part2/page_rank/page_rank.cpp
	rm -rf HW3_0816153.zip temp
	mkdir temp
	cp part1/cg_impl.c temp/cg_impl.c
	cp part2/breadth_first_search/bfs.cpp temp/bfs.cpp 
	cp part2/page_rank/page_rank.cpp temp/page_rank.cpp
	cd temp && zip HW3_0816153.zip cg_impl.c bfs.cpp page_rank.cpp
	cd temp && mv HW3_0816153.zip ../HW3_0816153.zip
	rm -rf temp
	printf "\033[92mZip Complete!\033[0m\n"

test: HW3_0816153.zip
	test_hw3

clean:
	rm -rf HW3_0816153.zip temp
