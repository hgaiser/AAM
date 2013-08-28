/*
 * DetectFace.cpp
 *
 *  Created on: 16 mei 2013
 *      Author: hansgaiser
 */

#include "DetectFace.h"

#include <functional>
#include <iomanip>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/freeglut.h>
#endif

namespace eva
{

/**
 * Constructor.
 */
DetectFace::DetectFace(const char * workingDir) :
		m_workingDir(workingDir),
		m_configPath(std::string(workingDir) + "/config/"),
		m_modelLoaded(false)
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
#ifdef HAVE_CUDA
		std::cout << ("[DetectFace] CUDA is enabled.");
		cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

		if (m_faceCascadeGpu.load(m_workingDir + "/cascade/" + m_config.faceCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade for Gpu; " << m_config.noseCascadeName.c_str() << std::endl;
		if (m_noseCascadeGpu.load(m_workingDir + "/cascade/" + m_config.noseCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade for Gpu; " << m_config.noseCascadeName.c_str() << std::endl;
#endif
	}
	else
	{
		if (m_faceCascade.load(m_workingDir + "/cascade/" + m_config.faceCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade; " << m_config.faceCascadeName.c_str() << std::endl;
		if (m_noseCascade.load(m_workingDir + "/cascade/" + m_config.noseCascadeName) == false)
			std::cerr << "[DetectFace] Could not load frontal face classifier cascade; " << m_config.noseCascadeName.c_str() << std::endl;
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
	if (matio.open(m_workingDir + "/model/" + m_model.modelName.c_str(), "r") == false)
	{
		std::cerr << "[DetectFace] Failed to load model, can't find it." << std::endl;
		return;
	}

	std::vector<MatlabIOContainer> variables;
	variables = matio.read();
	matio.close();

	std::vector<std::vector<MatlabIOContainer> > aam;
	aam = variables[0].data<std::vector<std::vector<MatlabIOContainer> > >();

	m_model.size = matio.find<cv::Mat>(aam[0], "size");

	m_model.shape_mean = matio.find<cv::Mat>(aam[0], "shape_mean");
	m_model.shape_mean.convertTo(m_model.shape_mean, MAT_TYPE(1));
	m_model.shape_vectors = matio.find<cv::Mat>(aam[0], "shape_ev");
	m_model.shape_vectors.convertTo(m_model.shape_vectors, MAT_TYPE(1));
	m_model.shape_transform = matio.find<cv::Mat>(aam[0], "shape_gt");
	m_model.shape_transform.convertTo(m_model.shape_transform, MAT_TYPE(1));
	m_model.shape_mesh = matio.find<cv::Mat>(aam[0], "shape_mesh");

	m_model.app_mean = matio.find<cv::Mat>(aam[0], "app_mean");
	m_model.app_mean.convertTo(m_model.app_mean, MAT_TYPE(3));
	//m_model.app_vectors = matio.find<cv::Mat>(aam[0], "app_ev");
	//m_model.app_vectors.convertTo(m_model.app_vectors, MAT_TYPE(1));
	// m_model.gradient = matio.find<cv::Mat>(aam[0], "gradient");

	// m_model.steepest_descent = matio.find<cv::Mat>(aam[0], "SD");
	// m_model.H = matio.find<cv::Mat>(aam[0], "H");
	// m_model.invH = matio.find<cv::Mat>(aam[0], "invH");
	std::vector<MatlabIOContainer> R = matio.find<std::vector<MatlabIOContainer> >(aam[0], "R");
	for (int i = 0; i < R.size(); i++)
	{
		m_model.R.push_back(R[i].data<cv::Mat>());
		m_model.R[i].convertTo(m_model.R[i], MAT_TYPE(3));
	}

	/*m_model.R = matio.find<cv::Mat>(aam[0], "R");
	m_model.R.convertTo(m_model.R, MAT_TYPE(1));
	m_model.eigen_R = new Eigen::Map<Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> >((float *)m_model.R.data, m_model.R.rows, m_model.R.cols);*/

	m_model.warp_map = matio.find<cv::Mat>(aam[0], "warp_map");

	m_model.width = m_model.size.at<double>(0, 1);
	m_model.height = m_model.size.at<double>(0, 0);

	m_model.no_pixels = matio.find<double>(aam[0], "no_pixels");

	m_modelLoaded = true;
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
	m_config.iterationThreshold = 0.001;
	m_config.maxIteration = 10;
	m_config.translationIteration = 5;

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
			if ((node = tmp->FirstChildElement("iterationThreshold")))
				m_config.iterationThreshold = atof(node->GetText());
			if ((node = tmp->FirstChildElement("translation_iteration")))
				m_config.translationIteration = atoi(node->GetText());
			if ((node = tmp->FirstChildElement("max_iteration")))
				m_config.maxIteration = atoi(node->GetText());
#ifdef HAVE_CUDA
			if (cv::gpu::getCudaEnabledDeviceCount() && (node = tmp->FirstChildElement("cuda")))
				m_config.cudaEnabled = strcmp(node->GetText(), "true") == 0;
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
		cv::circle(image, cv::Point(points.at<TYPE>(i, 1), points.at<TYPE>(i, 0)), 1, color, 2, CV_AA);
}

/**
 * Draws mesh on image with color.
 */
void DetectFace::drawMesh(cv::Mat & image, cv::Mat points, cv::Scalar color, cv::Mat mesh)
{
	for (int t = 0; t < mesh.rows; t++)
	{
		cv::line(image, cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 0), 1), points.at<TYPE>(mesh.at<int16_t>(t, 0), 0)), cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 1), 1), points.at<TYPE>(mesh.at<int16_t>(t, 1), 0)), color, 1);
		cv::line(image, cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 1), 1), points.at<TYPE>(mesh.at<int16_t>(t, 1), 0)), cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 2), 1), points.at<TYPE>(mesh.at<int16_t>(t, 2), 0)), color, 1);
		cv::line(image, cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 2), 1), points.at<TYPE>(mesh.at<int16_t>(t, 2), 0)), cv::Point(points.at<TYPE>(mesh.at<int16_t>(t, 0), 1), points.at<TYPE>(mesh.at<int16_t>(t, 0), 0)), color, 1);
	}

	for (int i = 0; i < points.rows; i++)
		cv::circle(image, cv::Point(points.at<TYPE>(i, 1), points.at<TYPE>(i, 0)), 1, cv::Scalar(0), 2, CV_AA);
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

GLuint textureID;
void loadImageToTexture(cv::Mat image)
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.ptr());
}

static const TYPE fInv640 = 1.0f / 640.0f;
static const TYPE fInv480 = 1.0f / 480.0f;
void warpImageGL(cv::Mat curr_points, cv::Mat shape_mesh, cv::Mat shape_mean)
{
    // Clear background
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_TRIANGLES);
	for (int t = 0; t < shape_mesh.rows; t++)
	{
		glTexCoord2d(curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 0), 1) * fInv640, curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 0), 0) * fInv480);
		glVertex2d  (shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 0), 1),            shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 0), 0));

		glTexCoord2d(curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 1), 1) * fInv640, curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 1), 0) * fInv480);
		glVertex2d  (shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 1), 1),            shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 1), 0));

		glTexCoord2d(curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 2), 1) * fInv640, curr_points.at<TYPE>(shape_mesh.at<int16_t>(t, 2), 0) * fInv480);
		glVertex2d  (shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 2), 1),            shape_mean.at<TYPE>(shape_mesh.at<int16_t>(t, 2), 0));
	}
    glEnd();

    // Flush pipeline
    glFlush();
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
		TYPE i = warp_map.at<TYPE>(w, 0) - 1;
		TYPE j = warp_map.at<TYPE>(w, 1) - 1;
		TYPE t = warp_map.at<TYPE>(w, 2) - 1;
		TYPE alpha = warp_map.at<TYPE>(w, 3);
		TYPE beta = warp_map.at<TYPE>(w, 4);
		TYPE gamma = warp_map.at<TYPE>(w, 5);

		cv::Mat srcTri(3, 2, MAT_TYPE(1));
		src.row(tris.at<int16_t>(t, 0)).copyTo(srcTri.row(0));
		src.row(tris.at<int16_t>(t, 1)).copyTo(srcTri.row(1));
		src.row(tris.at<int16_t>(t, 2)).copyTo(srcTri.row(2));

		cv::Mat projection = alpha * srcTri.row(0) + beta * srcTri.row(1) + gamma * srcTri.row(2);

		cv::Mat floor_prj;
		projection.copyTo(floor_prj);
		floor_prj.at<TYPE>(0, 0) = floor(floor_prj.at<TYPE>(0, 0));
		floor_prj.at<TYPE>(0, 1) = floor(floor_prj.at<TYPE>(0, 1));

		cv::Mat ceil_prj = floor_prj + 1.0;
		cv::Mat diff_prj = projection - floor_prj;

		cv::Mat neighbors[3];
		neighbors[0] = cv::Mat(2, 2, MAT_TYPE(1));
		neighbors[0].at<TYPE>(0, 0) = channels[0].at<uint8_t>(floor_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[0].at<TYPE>(0, 1) = channels[0].at<uint8_t>(floor_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));
		neighbors[0].at<TYPE>(1, 0) = channels[0].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[0].at<TYPE>(1, 1) = channels[0].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));

		neighbors[1] = cv::Mat(2, 2, MAT_TYPE(1));
		neighbors[1].at<TYPE>(0, 0) = channels[1].at<uint8_t>(floor_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[1].at<TYPE>(0, 1) = channels[1].at<uint8_t>(floor_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));
		neighbors[1].at<TYPE>(1, 0) = channels[1].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[1].at<TYPE>(1, 1) = channels[1].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));

		neighbors[2] = cv::Mat(2, 2, MAT_TYPE(1));
		neighbors[2].at<TYPE>(0, 0) = channels[2].at<uint8_t>(floor_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[2].at<TYPE>(0, 1) = channels[2].at<uint8_t>(floor_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));
		neighbors[2].at<TYPE>(1, 0) = channels[2].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), floor_prj.at<TYPE>(0, 1));
		neighbors[2].at<TYPE>(1, 1) = channels[2].at<uint8_t>(ceil_prj.at<TYPE>(0, 0), ceil_prj.at<TYPE>(0, 1));

		cv::Mat y(1, 2, MAT_TYPE(1));
		y.at<TYPE>(0, 0) = 1 - diff_prj.at<TYPE>(0, 0);
		y.at<TYPE>(0, 1) = diff_prj.at<TYPE>(0, 0);

		cv::Mat x(2, 1, MAT_TYPE(1));
		x.at<TYPE>(0, 0) = 1 - diff_prj.at<TYPE>(0, 1);
		x.at<TYPE>(1, 0) = diff_prj.at<TYPE>(0, 1);

		warpedChannels[0].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[0] * x).at<TYPE>(0,0));
		warpedChannels[1].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[1] * x).at<TYPE>(0,0));
		warpedChannels[2].at<uint8_t>(i, j) = uint8_t(cv::Mat(y * neighbors[2] * x).at<TYPE>(0,0));
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
	cv::Mat base(2, 3, MAT_TYPE(1));
	cv::Mat warped(2, 3, MAT_TYPE(1));

	for (int i = 0; i < 3; i++)
	{
		base.at<TYPE>(0, i) = aam.shape_mean.at<TYPE>(t.at<int16_t>(0, i), 0);
		base.at<TYPE>(1, i) = aam.shape_mean.at<TYPE>(t.at<int16_t>(0, i), 1);

		warped.at<TYPE>(0, i) = cv::Mat(base.at<TYPE>(0, i) + aam.shape_transform.row(t.at<int16_t>(0, i)) * q.t()).at<TYPE>(0,0);
		warped.at<TYPE>(1, i) = cv::Mat(base.at<TYPE>(1, i) + aam.shape_transform.row(np + t.at<int16_t>(0, i)) * q.t()).at<TYPE>(0,0);
	}

	TYPE den = (base.at<TYPE>(0,1) - base.at<TYPE>(0,0)) * (base.at<TYPE>(1,2) - base.at<TYPE>(1,0)) - (base.at<TYPE>(1,1) - base.at<TYPE>(1,0)) * (base.at<TYPE>(0,2) - base.at<TYPE>(0,0));
	TYPE alpha = (-base.at<TYPE>(0,0) * (base.at<TYPE>(1,2) - base.at<TYPE>(1,0)) + base.at<TYPE>(1,0) * (base.at<TYPE>(0,2) - base.at<TYPE>(0,0))) / den;
	TYPE beta  = (-base.at<TYPE>(1,0) * (base.at<TYPE>(0,1) - base.at<TYPE>(0,0)) + base.at<TYPE>(0,0) * (base.at<TYPE>(1,1) - base.at<TYPE>(1,0))) / den;

	// We start with the translation component
	TYPE a1 = warped.at<TYPE>(0,0) + (warped.at<TYPE>(0,1) - warped.at<TYPE>(0,0)) * alpha + (warped.at<TYPE>(0,2) - warped.at<TYPE>(0,0)) * beta;
	TYPE a4 = warped.at<TYPE>(1,0) + (warped.at<TYPE>(1,1) - warped.at<TYPE>(1,0)) * alpha + (warped.at<TYPE>(1,2) - warped.at<TYPE>(1,0)) * beta;

	alpha = (base.at<TYPE>(1,2) - base.at<TYPE>(1,0)) / den;
	beta  = (base.at<TYPE>(1,0) - base.at<TYPE>(1,1)) / den;

	// Relationships between original x coordinate and warped x and y coordinates
	TYPE a2 = (warped.at<TYPE>(0,1) - warped.at<TYPE>(0,0)) * alpha + (warped.at<TYPE>(0,2) - warped.at<TYPE>(0,0)) * beta;
	TYPE a5 = (warped.at<TYPE>(1,1) - warped.at<TYPE>(1,0)) * alpha + (warped.at<TYPE>(1,2) - warped.at<TYPE>(1,0)) * beta;

	alpha = (base.at<TYPE>(0,1) - base.at<TYPE>(0,0)) / den;
	beta  = (base.at<TYPE>(0,0) - base.at<TYPE>(0,2)) / den;

	// Relationships between original y coordinate and warped x and y coordinates
	TYPE a3 = (warped.at<TYPE>(0,2) - warped.at<TYPE>(0,0)) * alpha + (warped.at<TYPE>(0,1) - warped.at<TYPE>(0,0)) * beta;
	TYPE a6 = (warped.at<TYPE>(1,2) - warped.at<TYPE>(1,0)) * alpha + (warped.at<TYPE>(1,1) - warped.at<TYPE>(1,0)) * beta;

	// Store in matrix form
	// To be used in this way:
	// shape * A + tr ==> N(shape, q)
	tr = cv::Mat(1, 2, MAT_TYPE(1));
	tr.at<TYPE>(0, 0) = a1;
	tr.at<TYPE>(0, 1) = a4;

	A = cv::Mat(2, 2, MAT_TYPE(1));
	A.at<TYPE>(0, 0) = a2;
	A.at<TYPE>(0, 1) = a5;
	A.at<TYPE>(1, 0) = a3;
	A.at<TYPE>(1, 1) = a6;
}

cv::Mat warp_composition(DetectFace::Model aam, cv::Mat d_s0)
{
	cv::Mat result = cv::Mat::zeros(aam.curr_points.size(), MAT_TYPE(1));
	cv::Mat nt = cv::Mat::zeros(aam.shape_mean.rows, 1, CV_8UC1);
	for (int t = 0; t < aam.shape_mesh.rows; t++)
	{
		cv::Mat tr = aam.shape_mesh.row(t);

		nt.at<uint8_t>(tr.at<int16_t>(0, 0), 0)++;
		nt.at<uint8_t>(tr.at<int16_t>(0, 1), 0)++;
		nt.at<uint8_t>(tr.at<int16_t>(0, 2), 0)++;

		for (int k = 0; k < 3; k++)
		{
			cv::Mat t2;
			tr.copyTo(t2);

			t2.at<int16_t>(0, 0) = tr.at<int16_t>(0, k);
			t2.at<int16_t>(0, k) = tr.at<int16_t>(0, 0);

			TYPE i1 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 0), 0);
			TYPE j1 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 0), 1);
			TYPE i2 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 1), 0);
			TYPE j2 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 1), 1);
			TYPE i3 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 2), 0);
			TYPE j3 = aam.shape_mean.at<TYPE>(t2.at<int16_t>(0, 2), 1);

			TYPE i_coord = d_s0.at<TYPE>(t2.at<int16_t>(0, 0), 0);
			TYPE j_coord = d_s0.at<TYPE>(t2.at<int16_t>(0, 0), 1);

			TYPE den = (i2 - i1) * (j3 - j1) - (j2 - j1) * (i3 - i1);
			TYPE alpha = ((i_coord - i1) * (j3 - j1) - (j_coord - j1) * (i3 - i1)) / den;
			TYPE beta = ((j_coord - j1) * (i2 - i1) - (i_coord - i1) * (j2 - j1)) / den;

			result.row(t2.at<int16_t>(0, 0)) += (alpha * (aam.curr_points.row(t2.at<int16_t>(0, 1)) - aam.curr_points.row(t2.at<int16_t>(0, 0))) +
					beta * (aam.curr_points.row(t2.at<int16_t>(0, 2)) - aam.curr_points.row(t2.at<int16_t>(0, 0))));
		}
	}

	cv::divide(result.col(0), nt, result.col(0));
	cv::divide(result.col(1), nt, result.col(1));
	return result + aam.curr_points;
}

/**
 * Matches the AAM model to the current image.
 */
void DetectFace::matchModel(cv::Mat image)
{
	//cv::Mat im;
	//cv::cvtColor(image, im, CV_BGR2RGB);
	loadImageToTexture(image);

	TYPE err = INFINITY;

	for (int it = 0; it < m_config.maxIteration; it++)
	{
#if 0

		cv::Mat warped_im;
		cv::cvtColor(image, warped_im, CV_BGR2RGB);

		//ros::Time s = ros::Time::now();
		warped_im = warpImage(warped_im, m_model.curr_points, m_model.shape_mesh, m_model.warp_map);
		//std::cout << "Warping took: " << (ros::Time::now() - s).toSec() << std::endl;

		warped_im = warped_im(cv::Rect(0, 0, m_model.width, m_model.height));

#else

		cv::Mat warped_im(m_model.height, m_model.width, CV_8UC3);

		// Why does this not work?
		//fnDisplayFunc = std::bind(warpImageGL, m_model.curr_points, m_model.shape_mesh, m_model.shape_mean);
		//glutMainLoopEvent();

		// Call directly
		warpImageGL(m_model.curr_points, m_model.shape_mesh, m_model.shape_mean);

		// Read warped image
		glPixelStorei(GL_PACK_ALIGNMENT, (warped_im.step & 3) ? 1 : 4);
		glPixelStorei(GL_PACK_ROW_LENGTH, warped_im.step / warped_im.elemSize());
		glReadPixels(0, 0, warped_im.cols, warped_im.rows, GL_BGR, GL_UNSIGNED_BYTE, warped_im.data);

#endif
		/*cv::Mat disp_warped_im, disp_im;
		image.copyTo(disp_im);
		drawMesh(disp_im, m_model.curr_points, cv::Scalar(255, 0, 0), m_model.shape_mesh);
		//cv::cvtColor(warped_im, disp_warped_im, CV_RGB2BGR);
		warped_im.copyTo(disp_warped_im);
		drawMesh(disp_warped_im, m_model.shape_mean, cv::Scalar(255, 0, 0), m_model.shape_mesh);
		cv::imshow("Normal Image", disp_im);
		cv::imshow("Warped Image", disp_warped_im);
		cv::waitKey();
		//std::cout << "Warping took: " << (ros::Time::now() - s).toSec() << std::endl;*/

		warped_im = warped_im(cv::Rect(0, 0, m_model.width, m_model.height));
		warped_im.convertTo(warped_im, MAT_TYPE(1), 1.0/255.0);

		// calculate the error image, is there a better way perhaps?
		cv::Mat err_im = warped_im - m_model.app_mean;

		// the first few iterations only translate the model on top of the face, the other iterations also warps the model
		if (m_model.trans_it)
		{
			m_model.trans_it--;

			// transformation parameters
			cv::Mat delta_q(4, 1, MAT_TYPE(1));
			for (int i = 0; i < 4; i++)
				delta_q.at<TYPE>(i, 0) = cv::sum(cv::sum(m_model.R[i].mul(err_im)))[0];

			cv::Mat A, tr;
			to_affine(m_model, -delta_q.t(), A, tr);

			// calculate the delta shape
			cv::Mat delta_shape = (m_model.shape_mean * A);
			delta_shape.col(0) += tr.at<TYPE>(0, 0);
			delta_shape.col(1) += tr.at<TYPE>(0, 1);
			delta_shape -= m_model.shape_mean;

			m_model.curr_points += delta_shape;
		}
		else
		{
			// do transformation and warping
			cv::Mat delta_qp(m_model.R.size(), 1, MAT_TYPE(1));
			for (int i = 0; i < m_model.R.size(); i++)
				delta_qp.at<TYPE>(i, 0) = cv::sum(cv::sum(m_model.R[i].mul(err_im)))[0];

			cv::Mat d_s0(m_model.shape_mean.size(), MAT_TYPE(1));
			cv::Mat s;
			cv::reduce(m_model.shape_vectors * delta_qp.rowRange(4, delta_qp.rows), s, 1, CV_REDUCE_SUM);
			s.rowRange(0, d_s0.rows).copyTo(d_s0.col(0));
			s.rowRange(d_s0.rows, s.rows).copyTo(d_s0.col(1));
			d_s0 = m_model.shape_mean - d_s0;

			cv::Mat A, tr;
			to_affine(m_model, -delta_qp.rowRange(0, 4).t(), A, tr);

			d_s0 *= A;
			d_s0.col(0) += tr.at<TYPE>(0, 0);
			d_s0.col(1) += tr.at<TYPE>(0, 1);

			m_model.curr_points = warp_composition(m_model, d_s0);
		}

		// calculate the error, if it is very low we can stop iterating
		cv::Mat err_im_squared;
		cv::pow(err_im, 2, err_im_squared);
		TYPE curr_err = cv::sum(err_im_squared)[0] / m_model.no_pixels;
		//std::cout << "err: " << curr_err << std::endl;

		if (err - curr_err < m_config.iterationThreshold)
			break;

		err = curr_err;
	}
}

/**
 * Finds a face in the image to initialize the AAM with and calls the matchModel function.
 */
void DetectFace::processImage(cv::Mat image)
{
	if (m_config.scale != 1.0)
		cv::resize(image, image, cv::Size(image.cols * m_config.scale, image.rows * m_config.scale), 0, 0, cv::INTER_LINEAR);

	if (m_modelLoaded)
	{
		cv::Mat grayImage;

		cv::cvtColor(image, grayImage, CV_RGB2GRAY);

#ifdef HAVE_CUDA
		cv::gpu::GpuMat imageGpu;
		if (m_config.cudaEnabled)
			imageGpu.upload(grayImage);
#endif


		//create a vector array to store the face found
		std::vector<cv::Rect> faces;

		//find faces and store them in the vector array
#ifdef HAVE_CUDA
		if (m_config.cudaEnabled)
		{
			cv::gpu::GpuMat facesGpu;
			m_faceCascadeGpu.findLargestObject = true;
			//m_faceCascadeGpu.visualizeInPlace = true;
			int nrdetect = m_faceCascadeGpu.detectMultiScale(imageGpu, facesGpu, 1.1, 3, cv::Size(30,30));

			cv::Mat facesDownloaded;
			facesGpu.colRange(0, nrdetect).download(facesDownloaded);

			faces.insert(faces.end(), &facesDownloaded.ptr<cv::Rect>()[0], &facesDownloaded.ptr<cv::Rect>()[nrdetect]);
		}
		else
#endif
			m_faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));

		if (m_model.curr_points.empty())
		{
			// found a face?
			if (faces.size())
			{
				//cv::rectangle(image, faces[0], cv::Scalar(255, 0, 0));

				// find a nose
				std::vector<cv::Rect> noses;

#ifdef HAVE_CUDA
				if (m_config.cudaEnabled)
				{
					cv::gpu::GpuMat nosesGpu;
					m_faceCascadeGpu.findLargestObject = true;
					//m_faceCascadeGpu.visualizeInPlace = true;
					int nrdetect = m_noseCascadeGpu.detectMultiScale(imageGpu, nosesGpu, 1.1, 3, cv::Size(30,30));

					cv::Mat nosesDownloaded;
					nosesGpu.colRange(0, nrdetect).download(nosesDownloaded);

					noses.insert(noses.end(), &nosesDownloaded.ptr<cv::Rect>()[0], &nosesDownloaded.ptr<cv::Rect>()[nrdetect]);
				}
				else
#endif
					m_noseCascade.detectMultiScale(grayImage, noses, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, cv::Size(30,30), cv::Size(faces[0].width / 2, faces[0].height / 2));

				if (noses.size())
				{
					//cv::rectangle(image, noses[0], cv::Scalar(0, 0, 255));

					// center the face region around the nose
					faces[0].x = noses[0].x + noses[0].width / 2 - faces[0].width / 2;
					faces[0].y = noses[0].y + noses[0].height / 2 - faces[0].height / 2;

					//cv::rectangle(image, faces[0], cv::Scalar(0, 255, 0));

					// calculate the scale and initialize the points and the offset
					TYPE scale = std::min(TYPE(faces[0].width) / m_model.width, TYPE(faces[0].height) / m_model.height);
					cv::Point offset = cv::Point(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);
					m_model.shape_mean.copyTo(m_model.curr_points);

					// scale the mean points and place them on the center of the found face
					TYPE meanx = cv::mean(m_model.curr_points.col(1))[0];
					TYPE meany = cv::mean(m_model.curr_points.col(0))[0];
					m_model.curr_points.col(1) -= meanx;
					m_model.curr_points.col(0) -= meany;

					m_model.curr_points *= scale;

					double minx;
					double miny;
					cv::minMaxLoc(m_model.curr_points.col(1), &minx);
					cv::minMaxLoc(m_model.curr_points.col(0), &miny);
					m_model.curr_points.col(1) -= (minx - faces[0].x);
					m_model.curr_points.col(0) -= (miny - faces[0].y);

					m_model.trans_it = m_config.translationIteration;
				}
			}
		}
		else
		{
			//ros::Time s = ros::Time::now();
			matchModel(image);
			//std::cout << "Entire matching took: " << (ros::Time::now() - s).toSec() << std::endl;

			if (m_config.showImages)
			{
				//drawPoints(image, m_model.curr_points, cv::Scalar(0, 255, 0));
				//drawMesh(image, m_model.curr_points, cv::Scalar(0, 255, 0), m_model.shape_mesh);
			}
		}
	}
}

void displayFunc()
{
}

/**
 * Main loop.
 */
void DetectFace::spin()
{
    cv::Mat image = cv::imread(m_workingDir + "/image.jpg");

    // Setup GLUT environment
    int argc = 0;
    glutInit(&argc, 0);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(m_model.width, m_model.height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Warped Face");
    glutDisplayFunc(displayFunc);
    // Setup OpenGL 2D rendering
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, m_model.width, 0.0f, m_model.height, 0.0f, 1.0f); // Upside down image (for OpenCV)
    //glOrtho(0.0f, m_model.width, m_model.height, 0.0f, 0.0f, 1.0f); // Normal image
    // Load texture
    loadImageToTexture(image);

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

