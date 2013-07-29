#include "DetectFace.h"

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cerr << "Please provide a working directory (for example: ./AAM ../)." << std::endl;
		return 1;
	}

	eva::DetectFace df(argv[1]);
	df.spin();

	return 0;
}
