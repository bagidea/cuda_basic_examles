CC = /mnt/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.2/bin/nvcc.exe 
DB = /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
all: clean create_dir ex0 ex1 ex2 ex3 ex4
	@echo :::::::::::::::::::::::::::::
	@echo :::: All build completed ::::
	@echo :::::::::::::::::::::::::::::
ex0:
	@echo building... ex0
	@$(CC) Ex0.cu -o bin/Ex0
	@echo Done.
	@echo
ex1:
	@echo building... ex1
	@$(CC) Ex1.cu -o bin/Ex1
	echo Done.
	@echo
ex2:
	@echo building... ex2
	@$(CC) Ex2.cu -o bin/Ex2
	@echo Done.
	@echo
ex3:
	@echo building... ex3
	@$(CC) Ex3.cu -o bin/Ex3
	@echo Done.
	@echo
ex4:
	@echo building... ex4
	@$(CC) Ex4.cu -o bin/Ex4
	@echo Done.
	@echo
create_dir:
	@echo Initialized project GPU Parallel Programming.
	@mkdir bin
	@echo
run:
	@$(DB) bin/$(target)
clean:
	@echo Clean all.
	@rm -rf bin
