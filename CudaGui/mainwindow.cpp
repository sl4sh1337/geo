#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include "tiffio.h"

struct rgb_struct
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

extern "C" void kern(uint32* data, uint32 h, uint32 w, uint32* ans, int* kernel);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{

    ui->progressBar->setValue(0);
    TIFF* t= TIFFOpen(("img/old/" + ui->lineEdit->text().toStdString() + ".tif").data(), "r");
    if(t)
    {
        int* kernel = new int[9];

        kernel[0] = ui->lineEdit_0->text().toInt();
        kernel[1] = ui->lineEdit_1->text().toInt();
        kernel[2] = ui->lineEdit_2->text().toInt();
        kernel[3] = ui->lineEdit_3->text().toInt();
        kernel[4] = ui->lineEdit_4->text().toInt();
        kernel[5] = ui->lineEdit_5->text().toInt();
        kernel[6] = ui->lineEdit_6->text().toInt();
        kernel[7] = ui->lineEdit_7->text().toInt();
        kernel[8] = ui->lineEdit_8->text().toInt();

        uint32 w, h;
        TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
        uint32* img = (uint32*)_TIFFmalloc(w * h * sizeof(uint32));
        TIFFReadRGBAImageOriented(t, w, h, img, ORIENTATION_TOPLEFT);

        TIFFClose(t);
        ui->progressBar->setValue(33);
        uint32* new_img = new unsigned int[(h - 2) * (w - 2)];

        kern(img, h, w, new_img, kernel);
        ui->progressBar->setValue(66);
        _TIFFfree(img);
        QImage im(w - 2, h  - 2, QImage::Format_RGB32);
        for (uint32 i = 0; i < h - 2; i++)
        {
            for (uint32 j = 0; j < w - 2; j++)
            {
                auto val = (rgb_struct*)&new_img[i * (w - 2) + j];
                im.setPixel(j, i, qRgb(+val->red, +val->green, +val->blue));
            }
            std::cout << "\n";
        }
        im.save("img/new/new_" + ui->lineEdit->text() + ".tif", "tif", 0);

        delete [] new_img;
        delete [] kernel;
        ui->progressBar->setValue(100);
    }
}
