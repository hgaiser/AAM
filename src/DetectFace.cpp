/*
 * DetectFace.cpp
 *
 *  Created on: 16 mei 2013
 *      Author: hansgaiser
 */

#include "DetectFace.h"

namespace eva
{

/**
 * Constructor.
 */
DetectFace::DetectFace(const char * workingDir) :
		m_workingDir(workingDir),
		m_configPath(std::string(workingDir) + "/config/")
{
	loadConfig();

	if (m_config.faceCascadeName == "")
	{
		std::cerr << "[DetectFace] No face cascade file found." << std::endl;
		return;
	}

	if (m_config.noseCascadeName == "")
	{
		std::cerr << "[DetectFace] No nose cascade file found." << std::endl;
		return;
	}

	if (m_config.cudaEnabled)
	{
#ifdef WITH_CUDA
		std::cout << "[DetectFace] CUDA is enabled." << std::endl;
		cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

		if (m_faceCascadeGpu.load(std::string(workingDir) + "/cascade/" + m_config.cascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade for Gpu; " << m_config.cascadeName << std::endl;
		if (m_faceCascadeGpu.load(std::string(workingDir) + "/cascade/" + m_config.cascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade for Gpu; ", m_config.cascadeName << std::endl;
#endif
	}
	else
	{
		if (m_faceCascade.load(std::string(workingDir) + "/cascade/" + m_config.faceCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade; " << m_config.faceCascadeName << std::endl;
		if (m_noseCascade.load(std::string(workingDir) + "/cascade/" + m_config.noseCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade; " << m_config.noseCascadeName << std::endl;
	}

	loadModel();

    if (m_config.showImages)
    {
#ifndef __APPLE__
		cv::startWindowThread();
#endif
		cv::namedWindow("DetectFace", CV_WINDOW_AUTOSIZE);
    }

	std::cout << "[DetectFace] Successfully initialised." << std::endl;
}

/**
 * Loads the AAM model.
 */
void DetectFace::loadModel()
{
	MatlabIO matio;
	matio.open(std::string(m_workingDir) + "/model/" + m_model.modelName.c_str(), "r");

	std::vector<MatlabIOContainer> variables;
	variables = matio.read();
	matio.close();

	std::vector<std::vector<MatlabIOContainer> > aam;
	aam = variables[0].data<std::vector<std::vector<MatlabIOContainer> > >();

	m_model.size = matio.find<cv::Mat>(aam[0], "size");

	m_model.shape_mean = matio.find<cv::Mat>(aam[0], "shape_mean");
	m_model.shape_vectors = matio.find<cv::Mat>(aam[0], "shape_ev");
	m_model.shape_transform = matio.find<cv::Mat>(aam[0], "shape_gt");
	m_model.shape_mesh = matio.find<cv::Mat>(aam[0], "shape_mesh");

	m_model.app_mean = matio.find<cv::Mat>(aam[0], "app_mean");
	m_model.app_vectors = matio.find<cv::Mat>(aam[0], "app_ev");
	// m_model.gradient = matio.find<cv::Mat>(aam[0], "gradient");

	// m_model.steepest_descent = matio.find<cv::Mat>(aam[0], "SD");
	// m_model.H = matio.find<cv::Mat>(aam[0], "H");
	// m_model.invH = matio.find<cv::Mat>(aam[0], "invH");
	m_model.R = matio.find<cv::Mat>(aam[0], "R");

	m_model.warp_map = matio.find<cv::Mat>(aam[0], "warp_map");

	m_model.width = m_model.size.at<double>(0, 1);
	m_model.height = m_model.size.at<double>(0, 0);
}

/**
 * Loads the XML config file.
 */
bool DetectFace::loadConfig()
{
	// default values
	m_config.faceCascadeName = "frontalface_default.xml";
	m_config.noseCascadeName = "nose_default.xml";
	m_config.imageTopic = "/camera/rgb/image_color";
	m_config.showImages = true;
	m_config.cudaEnabled = false;
	m_config.scale = 1.0;

	m_model.modelName = "aam.mdl";

	TiXmlDocument doc(m_configPath + "detect_face.xml");
	if (doc.LoadFile() == false)
	{
		std::cout << "[DetectFace] Configuration file not found or failed to load. Assuming default values. " << doc.ErrorDesc() << std::endl;
		return false;
	}

	TiXmlElement *configNode = NULL;
	if ((configNode = doc.FirstChildElement("configuration")))
	{
		TiXmlElement *tmp = NULL;

		// cascade config
		if ((tmp = configNode->FirstChildElement("cascade")))
		{
			TiXmlElement *node = NULL;
			if ((node = tmp->FirstChildElement("facename")))
				m_config.faceCascadeName = node->GetText();
			if ((node = tmp->FirstChildElement("nosename")))
				m_config.noseCascadeName = node->GetText();
			if ((node = tmp->FirstChildElement("scale")))
				m_config.scale = atof(node->GetText());
#ifdef WITH_CUDA
			if (cv::gpu::getCudaEnabledDeviceCount() && (node = tmp->FirstChildElement("cuda")))
				m_config.m_cudaEnabled = strcmp(node->GetText(), "true") == 0;
#endif
		}

		// model config
		if ((tmp = configNode->FirstChildElement("model")))
		{
			TiXmlElement *node = NULL;
			if ((node = tmp->FirstChildElement("name")))
				m_model.modelName = node->GetText();
		}

		// image topic
		if ((tmp = configNode->FirstChildElement("image_topic")))
			m_config.imageTopic = tmp->GetText();
		if ((tmp = configNode->FirstChildElement("show_images")))
			m_config.showImages = strcmp(tmp->GetText(), "true") == 0;
	}

	return true;
}

/**
 * Draws points on image with color.
 */
void DetectFace::drawPoints(cv::Mat & image, cv::Mat points, cv::Scalar color)
{
	for (int i = 0; i < points.rows; i++)
		cv::circle(image, cv::Point(points.at<double>(i, 1), points.at<double>(i, 0)), 1, color, 2, CV_AA);
}

/**
 * Draws mesh on image with color.
 */
void DetectFace::drawMesh(cv::Mat & image, cv::Mat points, cv::Scalar color, cv::Mat mesh)
{
	for (int t = 0; t < mesh.rows; t++)
	{
		cv::line(image, cv::Point(points.at<double>(mesh.at<int16_t>(t, 0) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 0) - 1, 0)), cv::Point(points.at<double>(mesh.at<int16_t>(t, 1) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 1) - 1, 0)), color, 1, CV_AA);
		cv::line(image, cv::Point(points.at<double>(mesh.at<int16_t>(t, 1) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 1) - 1, 0)), cv::Point(points.at<double>(mesh.at<int16_t>(t, 2) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 2) - 1, 0)), color, 1, CV_AA);
		cv::line(image, cv::Point(points.at<double>(mesh.at<int16_t>(t, 2) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 2) - 1, 0)), cv::Point(points.at<double>(mesh.at<int16_t>(t, 0) - 1, 1), points.at<double>(mesh.at<int16_t>(t, 0) - 1, 0)), color, 1, CV_AA);
	}

	for (int i = 0; i < points.rows; i++)
		cv::circle(image, cv::Point(points.at<double>(i, 1), points.at<double>(i, 0)), 1, color, 2, CV_AA);
}

/**
 * Draws text in the upper left corner of the image.
 */
void DetectFace::drawText(cv::Mat & image, std::string text, cv::Scalar fontColor)
{
	// settings
	int fontFace = cv::FONT_HERSHEY_DUPLEX;
	double fontScale = 0.8;
 	int fontThickness = 2;
	cv::Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

	cv::Point org;
	org.x = 2 * fontSize.height;
	org.y = 2 * fontSize.height;

	putText(image, text, org, fontFace, fontScale, cv::Scalar(0,0,0), 5*fontThickness/2, 16);
	putText(image, text, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

/**
 * Warps image from src using the warp map and the triangles in tris.
 */
cv::Mat warpImage(cv::Mat image, cv::Mat src, cv::Mat tris, cv::Mat warp_map)
{
	cv::Mat warpedChannels[3];
	warpedChannels[0] = cv::Mat::zeros(image.size(), CV_8UC1);
	warpedChannels[1] = cv::Mat::zeros(image.size(), CV_8UC1);
	warpedChannels[2] = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat channels[3];
	cv::split(image, channels);

	for (int w = 0; w < warp_map.rows; w++)
	{
		double i = warp_map.at<double>(w, 0) - 1;
		double j = warp_map.at<double>(w, 1) - 1;
		double t = warp_map.at<double>(w, 2) - 1;
		double alpha = warp_map.at<double>(w, 3);
		double beta = warp_map.at<double>(w, 4);
		double gamma = warp_map.at<double>(w, 5);

		cv::Mat srcTri(3, 2, CV_64FC1);

		cv::Mat row0 = srcTri.row(0);
		cv::Mat row1 = srcTri.row(1);
		cv::Mat row2 = srcTri.row(2);

		src.row(tris.at<int16_t>(t, 0) - 1).copyTo(row0);
		src.row(tris.at<int16_t>(t, 1) - 1).copyTo(row1);
		src.row(tris.at<int16_t>(t, 2) - 1).copyTo(row2);

		cv::Mat projection = alpha * srcTri.row(0) + beta * srcTri.row(1) + gamma * srcTri.row(2);

		cv::Mat floor_prj;
		projection.copyTo(floor_prj);
		floor_prj.at<double>(0, 0) = floor(floor_prj.at<double>(0, 0));
		floor_prj.at<double>(0, 1) = floor(floor_prj.at<double>(0, 1));

		cv::Mat ceil_prj = floor_prj + 1.0;
		cv::Mat diff_prj = projection - floor_prj;

		cv::Mat neighbors[3];
		neighbors[0] = cv::Mat(2, 2, CV_64FC1);
		neighbors[0].at<double>(0, 0) = channels[0].at<uint8_t>(floor_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[0].at<double>(0, 1) = channels[0].at<uint8_t>(floor_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));
		neighbors[0].at<double>(1, 0) = channels[0].at<uint8_t>(ceil_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[0].at<double>(1, 1) = channels[0].at<uint8_t>(ceil_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));

		neighbors[1] = cv::Mat(2, 2, CV_64FC1);
		neighbors[1].at<double>(0, 0) = channels[1].at<uint8_t>(floor_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[1].at<double>(0, 1) = channels[1].at<uint8_t>(floor_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));
		neighbors[1].at<double>(1, 0) = channels[1].at<uint8_t>(ceil_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[1].at<double>(1, 1) = channels[1].at<uint8_t>(ceil_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));

		neighbors[2] = cv::Mat(2, 2, CV_64FC1);
		neighbors[2].at<double>(0, 0) = channels[2].at<uint8_t>(floor_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[2].at<double>(0, 1) = channels[2].at<uint8_t>(floor_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));
		neighbors[2].at<double>(1, 0) = channels[2].at<uint8_t>(ceil_prj.at<double>(0, 0), floor_prj.at<double>(0, 1));
		neighbors[2].at<double>(1, 1) = channels[2].at<uint8_t>(ceil_prj.at<double>(0, 0), ceil_prj.at<double>(0, 1));

		cv::Mat y(1, 2, CV_64FC1);
		y.at<double>(0, 0) = 1 - diff_prj.at<double>(0, 0);
		y.at<double>(0, 1) = diff_prj.at<double>(0, 0);

		cv::Mat x(2, 1, CV_64FC1);
		x.at<double>(0, 0) = 1 - diff_prj.at<double>(0, 1);
		x.at<double>(1, 0) = diff_prj.at<double>(0, 1);

		warpedChannels[0].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[0] * x).at<double>(0,0));
		warpedChannels[1].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[1] * x).at<double>(0,0));
		warpedChannels[2].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[2] * x).at<double>(0,0));
	}

	cv::Mat warped_im;
	cv::merge(warpedChannels, 3, warped_im);
	return warped_im;
}

/**
 * Computes the affine transformation corresponding to parameters q. A and tr represent the affine transformations.
 * Used as v_new = n * A + t.
 */
void to_affine(DetectFace::Model aam, cv::Mat q, cv::Mat & A, cv::Mat & tr)
{
	uint32_t np = aam.shape_mean.rows;
	cv::Mat t = aam.shape_mesh.row(0);
	cv::Mat base(2, 3, CV_64FC1);
	cv::Mat warped(2, 3, CV_64FC1);

	for (int i = 0; i < 3; i++)
	{
		base.at<double>(0, i) = aam.shape_mean.at<double>(t.at<int16_t>(0, i) - 1, 0);
		base.at<double>(1, i) = aam.shape_mean.at<double>(t.at<int16_t>(0, i) - 1, 1);

		warped.at<double>(0, i) = cv::Mat(base.at<double>(0, i) + aam.shape_transform.row(t.at<int16_t>(0, i) - 1) * q.t()).at<double>(0,0);
		warped.at<double>(1, i) = cv::Mat(base.at<double>(1, i) + aam.shape_transform.row(np + t.at<int16_t>(0, i) - 1) * q.t()).at<double>(0,0);
	}

	double den = (base.at<double>(0,1) - base.at<double>(0,0)) * (base.at<double>(1,2) - base.at<double>(1,0)) - (base.at<double>(1,1) - base.at<double>(1,0)) * (base.at<double>(0,2) - base.at<double>(0,0));
	double alpha = (-base.at<double>(0,0) * (base.at<double>(1,2) - base.at<double>(1,0)) + base.at<double>(1,0) * (base.at<double>(0,2) - base.at<double>(0,0))) / den;
	double beta  = (-base.at<double>(1,0) * (base.at<double>(0,1) - base.at<double>(0,0)) + base.at<double>(0,0) * (base.at<double>(1,1) - base.at<double>(1,0))) / den;

	// We start with the translation component
	double a1 = warped.at<double>(0,0) + (warped.at<double>(0,1) - warped.at<double>(0,0)) * alpha + (warped.at<double>(0,2) - warped.at<double>(0,0)) * beta;
	double a4 = warped.at<double>(1,0) + (warped.at<double>(1,1) - warped.at<double>(1,0)) * alpha + (warped.at<double>(1,2) - warped.at<double>(1,0)) * beta;

	alpha = (base.at<double>(1,2) - base.at<double>(1,0)) / den;
	beta  = (base.at<double>(1,0) - base.at<double>(1,1)) / den;

	// Relationships between original x coordinate and warped x and y coordinates
	double a2 = (warped.at<double>(0,1) - warped.at<double>(0,0)) * alpha + (warped.at<double>(0,2) - warped.at<double>(0,0)) * beta;
	double a5 = (warped.at<double>(1,1) - warped.at<double>(1,0)) * alpha + (warped.at<double>(1,2) - warped.at<double>(1,0)) * beta;

	alpha = (base.at<double>(0,1) - base.at<double>(0,0)) / den;
	beta  = (base.at<double>(0,0) - base.at<double>(0,2)) / den;

	// Relationships between original y coordinate and warped x and y coordinates
	double a3 = (warped.at<double>(0,2) - warped.at<double>(0,0)) * alpha + (warped.at<double>(0,1) - warped.at<double>(0,0)) * beta;
	double a6 = (warped.at<double>(1,2) - warped.at<double>(1,0)) * alpha + (warped.at<double>(1,1) - warped.at<double>(1,0)) * beta;

	// Store in matrix form
	// To be used in this way:
	// shape * A + tr ==> N(shape, q)
	tr = cv::Mat(1, 2, CV_64FC1);
	tr.at<double>(0, 0) = a1;
	tr.at<double>(0, 1) = a4;

	A = cv::Mat(2, 2, CV_64FC1);
	A.at<double>(0, 0) = a2;
	A.at<double>(0, 1) = a5;
	A.at<double>(1, 0) = a3;
	A.at<double>(1, 1) = a6;
}

/**
 * Matches the AAM model to the current image.
 */
void DetectFace::matchModel(cv::Mat image)
{
	cv::Mat im;
	cv::cvtColor(image, im, CV_BGR2RGB);

	//ros::Time s = ros::Time::now();
	im = warpImage(im, m_model.curr_points, m_model.shape_mesh, m_model.warp_map);
	//std::cout << "Warping took: " << (ros::Time::now() - s).toSec() << std::endl;

	im = im(cv::Rect(0, 0, m_model.width, m_model.height));
	im.convertTo(im, CV_64FC1, 1.0/255.0);
	cv::Mat channels[3];
	cv::split(im, channels);

	// calculate the error image, is there a better way perhaps?
	cv::Mat err_im(im.rows * im.cols * im.channels(), 1, CV_64FC1);
	uint32_t pix_per_channel = im.rows * im.cols;
	for (int c = 0; c < im.channels(); c++)
		for (int j = 0; j < im.cols; j++)
			err_im.rowRange(c * pix_per_channel + j * im.rows, c * pix_per_channel + (j + 1) * im.rows) = channels[c].col(j) - m_model.app_mean.rowRange(c * pix_per_channel + j * im.rows, c * pix_per_channel + (j + 1) * im.rows);

	// transformation parameters
	cv::Mat delta_q = m_model.R.rowRange(0, 4) * err_im;

	cv::Mat A, tr;
	to_affine(m_model, -delta_q.t(), A, tr);

	// calculate the delta shape
	cv::Mat delta_shape = (m_model.shape_mean * A);
	delta_shape.col(0) += tr.at<double>(0, 0);
	delta_shape.col(1) += tr.at<double>(0, 1);
	delta_shape -= m_model.shape_mean;

	m_model.curr_points += delta_shape;
}

/**
 * Finds a face in the image to initialize the AAM with and calls the matchModel function.
 */
//void DetectFace::processImage(const sensor_msgs::ImageConstPtr & imMsg)
 void DetectFace::processImage(cv::Mat image)
{
	//cv::Mat image = OpenCVTools::imageToMat(imMsg);

	if (m_config.scale != 1.0)
		cv::resize(image, image, cv::Size(image.cols * m_config.scale, image.rows * m_config.scale), 0, 0, cv::INTER_LINEAR);

	cv::Mat grayImage;

	cv::cvtColor(image, grayImage, CV_RGB2GRAY);

	//create a vector array to store the face found
	std::vector<cv::Rect> faces;

	//find faces and store them in the vector array
	if (m_config.cudaEnabled)
	{
#ifdef WITH_CUDA
		cv::gpu::GpuMat imageGpu(grayImage);
		cv::gpu::GpuMat facesGpu;
		m_faceCascadeGpu.findLargestObject = true;
		//m_faceCascadeGpu.visualizeInPlace = true;
        int nrdetect = m_faceCascadeGpu.detectMultiScale(imageGpu, facesGpu, 1.2, 3, cv::Size(30,30));

        cv::Mat facesDownloaded;
        facesGpu.colRange(0, nrdetect).download(facesDownloaded);

        faces.insert(faces.end(), &facesDownloaded.ptr<cv::Rect>()[0], &facesDownloaded.ptr<cv::Rect>()[nrdetect]);
#endif
	}
	else
		m_faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));

	if (m_model.curr_points.empty())
	{
		// found a face?
		if (faces.size())
		{
			// find a nose
			std::vector<cv::Rect> noses;
			m_noseCascade.detectMultiScale(grayImage, noses, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, cv::Size(30,30), cv::Size(faces[0].width / 2, faces[0].height / 2));
			if (noses.size() == 0)
				return;

			//cv::rectangle(image, faces[0], cv::Scalar(255, 0, 0));
			//cv::rectangle(image, noses[0], cv::Scalar(0, 0, 255));

			// center the face region around the nose
			faces[0].x = noses[0].x + noses[0].width / 2 - faces[0].width / 2;
			faces[0].y = noses[0].y + noses[0].height / 2 - faces[0].height / 2;

			//cv::rectangle(image, faces[0], cv::Scalar(0, 255, 0));

			// calculate the scale and initialize the points and the offset
			double scale = std::min(double(faces[0].width) / m_model.width, double(faces[0].height) / m_model.height);
			cv::Point offset = cv::Point(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);
			m_model.shape_mean.copyTo(m_model.curr_points);

			// scale the mean points and place them on the center of the found face
			double meanx = cv::mean(m_model.curr_points.col(1))[0];
			double meany = cv::mean(m_model.curr_points.col(0))[0];
			m_model.curr_points.col(1) -= meanx;
			m_model.curr_points.col(0) -= meany;

			m_model.curr_points *= scale;

			double minx;
			double miny;
			cv::minMaxLoc(m_model.curr_points.col(1), &minx);
			cv::minMaxLoc(m_model.curr_points.col(0), &miny);
			m_model.curr_points.col(1) -= (minx - faces[0].x);
			m_model.curr_points.col(0) -= (miny - faces[0].y);
		}
	}

	// can we match our model?
	if (m_model.curr_points.empty() == false)
		matchModel(image);
}

/**
 * Main loop.
 */
void DetectFace::spin()
{
	cv::Mat image = cv::imread(m_workingDir + "/image.jpg");

	//Start and end times
	time_t start,end;
	int counter = 0;
	time(&start);
	while (cv::waitKey(10) != 'q')
	{
		processImage(image);

		if (m_config.showImages && m_model.curr_points.empty() == false)
		{
			// calculate the fps
			time(&end);
			counter++;
			double sec = difftime(end, start);
			double fps = counter / sec;

			std::ostringstream ss;
			ss << std::setprecision(3) << std::setiosflags(std::ios::showpoint) << fps << " fps";

			cv::Mat draw_image;
			image.copyTo(draw_image);
			//drawPoints(draw_image, m_model.curr_points, cv::Scalar(0, 255, 0));
			drawMesh(draw_image, m_model.curr_points, cv::Scalar(0, 255, 0), m_model.shape_mesh);
			drawText(draw_image, ss.str(), cv::Scalar(0, 255, 0));

			cv::imshow("DetectFace", draw_image);
		}
		//cv::waitKey();
	}
}

} /* namespace eva */

