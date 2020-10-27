#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMovie>
#include "loaddialog.h"
#include "thread"
#include <QGraphicsScene>
#include <QDesktopWidget>

void MainWindow::update_tif(GeoTIFFWrapper* item, uint32 *new_data, int new_width, int new_height)
{
    if(buffered_tif_data != nullptr)
        delete[] buffered_tif_data;
    last_edited = item;
    buffered_tif_data = item->tif_data;
    buffered_tif_data_width = item->tif_data_width;
    buffered_tif_data_height = item->tif_data_height;

    item->tif_data = new_data;
    item->tif_data_width = new_width;
    item->tif_data_height = new_height;
    ++item->mods;
    viewer->updateItem(item);
    item->setEdited(true);
}

void MainWindow::on_actionUndo_triggered()
{
    delete[] last_edited->tif_data;
    last_edited->tif_data = buffered_tif_data;
    buffered_tif_data = nullptr;
    last_edited->tif_data_width = buffered_tif_data_width;
    last_edited->tif_data_height = buffered_tif_data_height;
    viewer->updateItem(last_edited);
    viewer->finishUpdating();
    ui->actionUndo->setEnabled(false);
    if(--last_edited->mods == 0)
    {
        last_edited->setEdited(false);
        ui->actionSave->setEnabled(false);
        ui->actionSave_2->setEnabled(false);
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->comboBox->addItem("0.25");
    ui->comboBox->addItem("0.5");
    ui->comboBox->addItem("0.75");
    ui->comboBox->addItem("1.25");
    ui->comboBox->addItem("1.5");
    ui->comboBox->addItem("1.75");
    ui->comboBox->addItem("2.0");
    ui->comboBox->setCurrentText("2.0");

    ui->comboBox_2->addItem(" ");
    ui->comboBox_2->addItem("Identity");
    ui->comboBox_2->addItem("Edge Detection 1");
    ui->comboBox_2->addItem("Edge Detection 2");
    ui->comboBox_2->addItem("Edge Detection 3");
    ui->comboBox_2->addItem("Sharpen");
    ui->comboBox_2->addItem("Box Blur");
    ui->comboBox_2->addItem("Gaussian Blur");

    viewer = new MultiViewer(this);
    connect(viewer, &MultiViewer::finished, this, &MainWindow::on_MultiViewerFinished);

    load = new LoadDialog(this);
    connect(this, &MainWindow::loaded, this, &MainWindow::closeLoad);

    connect(ui->listWidget->model(), &QAbstractItemModel::rowsMoved, this, &MainWindow::on_orderChanged);
    connect(viewer->scene, &QGraphicsScene::selectionChanged, this, &MainWindow::selectFiles);

    move((double)QApplication::desktop()->width() / 12, (double)QApplication::desktop()->height() / 12);
}

MainWindow::~MainWindow()
{
    delete ui;
    //for(int i = 0; i < opened_files.size(); ++i)
    //    delete opened_files[i];
}

void MainWindow::change_toggle()
{
    ui->actionImage_Window->toggle();
}

void MainWindow::kernelEachThr(int* kernel)
{
    for(int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->view->isSelected())
        {
            uint32* new_img = new unsigned int[(opened_files[i]->tif_data_width - 2) * (opened_files[i]->tif_data_height - 2)];

            kern(opened_files[i]->tif_data, opened_files[i]->tif_data_width, opened_files[i]->tif_data_height, new_img, kernel);
            update_tif(opened_files[i], new_img, opened_files[i]->tif_data_width - 2, opened_files[i]->tif_data_height - 2);
        }
    }

    emit loaded(false);
}

void MainWindow::openFileThr(const QString& filename)
{
    opened_files.push_back(new GeoTIFFWrapper(filename));
    viewer->addItem(opened_files.back());

    emit this->loaded(true);
}

void MainWindow::InterpolateThr(double s)
{
    s = std::sqrt(s);
    for(int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->view->isSelected())
        {

            int new_width = opened_files[i]->tif_data_width * s, new_height = opened_files[i]->tif_data_height * s;
            uint32* new_img = new uint32[new_width * new_height];

            resampling(opened_files[i]->tif_data, opened_files[i]->tif_data_width, opened_files[i]->tif_data_height, new_img, new_width, new_height, s);

            update_tif(opened_files[i], new_img, new_width, new_height);

        }
    }

    emit loaded(false);
}




void MainWindow::on_pushButton_clicked()
{
    int* kernel = new int[9];


    kernel[0] = ui->spinBox_1->value();
    kernel[1] = ui->spinBox_2->value();
    kernel[2] = ui->spinBox_3->value();
    kernel[3] = ui->spinBox_4->value();
    kernel[4] = ui->spinBox_5->value();
    kernel[5] = ui->spinBox_6->value();
    kernel[6] = ui->spinBox_7->value();
    kernel[7] = ui->spinBox_8->value();
    kernel[8] = ui->spinBox_9->value();

    std::thread thr(&MainWindow::kernelEachThr, this, kernel);
    thr.detach();

    load->exec();

}

void MainWindow::on_actionOpen_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open File", "", "GeoTIFF files (*.tif *.tiff)");

    if(!QFile::exists(filename))
        return;



    std::thread thr(&MainWindow::openFileThr, this, filename);
    thr.detach();

    load->exec();

}

void MainWindow::on_Interpolation_clicked()
{

    std::thread thr(&MainWindow::InterpolateThr, this, ui->comboBox->currentText().toDouble());
    thr.detach();

    load->exec();
}

void MainWindow::on_actionImage_Window_toggled(bool checked)
{
    if(checked)
        viewer->show();
    else
        viewer->hide();
}

void MainWindow::on_MultiViewerFinished(int r)
{
    ui->actionImage_Window->toggle();
}

void MainWindow::closeLoad(bool newFile)
{

    load->close();
    if(newFile)
    {
        ui->listWidget->addItem(opened_files.back());
        viewer->finishAdding();
    }
    else
    {
        ui->actionUndo->setEnabled(true);
        viewer->finishUpdating();
        if(ui->listWidget->selectedItems().size() == 1)
        {
            ui->actionSave->setEnabled(true);
            ui->actionSave_2->setEnabled(true);
        }
    }


    if(!ui->actionImage_Window->isChecked())
        ui->actionImage_Window->toggle();
}

void MainWindow::singularThr(int type)
{
    for(int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->view->isSelected())
        {
            uint32* new_img = new uint32[opened_files[i]->tif_data_width * opened_files[i]->tif_data_height];

            singular(opened_files[i]->tif_data, opened_files[i]->tif_data_width, opened_files[i]->tif_data_height, new_img, type);

            update_tif(opened_files[i], new_img, opened_files[i]->tif_data_width, opened_files[i]->tif_data_height);

        }
    }

    emit loaded(false);

}

void MainWindow::interSecThr(int type)
{
    GeoTIFFWrapper* first_image = nullptr;
    GeoTIFFWrapper* second_image = nullptr;
    qreal mainXLeft, mainXRight, mainYTop, mainYBottom;
    qreal intersecXLeft, intersecXRight, intersecYTop, intersecYBottom;
    for(int i = 0; i < ui->listWidget->count(); ++i)
    {
        if(ui->listWidget->item(i)->isSelected())
        {
            if(first_image == nullptr)
            {
                first_image = (GeoTIFFWrapper*)ui->listWidget->item(i);
                mainXLeft = first_image->view->x();
                mainXRight = first_image->view->x() + first_image->view->pixmap().width();
                mainYTop = first_image->view->y();
                mainYBottom = first_image->view->y() + first_image->view->pixmap().height();
            }
            else
            {
                second_image = (GeoTIFFWrapper*)ui->listWidget->item(i);
                intersecXLeft = std::max(mainXLeft, second_image->view->x());
                intersecXRight = std::min(mainXRight, second_image->view->x() + second_image->view->pixmap().width());
                intersecYTop = std::max(mainYTop, second_image->view->y());
                intersecYBottom = std::min(mainYBottom, second_image->view->y() + second_image->view->pixmap().height());

                if(intersecXLeft < intersecXRight && intersecYTop < intersecYBottom)
                {
                    uint32* new_img = new unsigned int[(first_image->tif_data_width) * (first_image->tif_data_height)];

                    intersec(first_image->tif_data, intersecXLeft - mainXLeft, intersecXRight - mainXLeft, intersecYTop - mainYTop, intersecYBottom - mainYTop, first_image->tif_data_width, first_image->tif_data_height,
                                            new_img,
                                            second_image->tif_data, intersecXLeft - second_image->view->x(), intersecXRight - second_image->view->x(), intersecYTop - second_image->view->y(), intersecYBottom - second_image->view->y(), second_image->tif_data_width, second_image->tif_data_height, type);

                    update_tif(first_image, new_img, first_image->tif_data_width, first_image->tif_data_height);
                }
            }
        }
    }

    emit loaded(false);
}

void MainWindow::on_abs_clicked()
{

    std::thread thr(&MainWindow::singularThr, this, 1);
    thr.detach();

    load->exec();
}

void MainWindow::on_round_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 2);
    thr.detach();

    load->exec();
}

void MainWindow::on_floor_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 3);
    thr.detach();

    load->exec();
}

void MainWindow::on_ceiling_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 4);
    thr.detach();

    load->exec();
}

void MainWindow::on_sqrt_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 5);
    thr.detach();

    load->exec();
}

void MainWindow::on_log_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 6);
    thr.detach();

    load->exec();
}

void MainWindow::on_exp_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 7);
    thr.detach();

    load->exec();
}

void MainWindow::on_cos_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 8);
    thr.detach();

    load->exec();
}

void MainWindow::on_sin_clicked()
{
    std::thread thr(&MainWindow::singularThr, this, 9);
    thr.detach();

    load->exec();
}

void MainWindow::on_orderChanged(const QModelIndex & sourceParent, int sourceStart, int sourceEnd, const QModelIndex & destinationParent, int destinationRow)
{
    for(int i = 0; i < ui->listWidget->count(); ++i)
    {
        GeoTIFFWrapper* itm = (GeoTIFFWrapper*)ui->listWidget->item(i);
        itm->view->setZValue(i);
    }
}

void MainWindow::selectFiles()
{
    ui->listWidget->blockSignals(true);
    for(int i = 0; i < opened_files.size(); ++i)
    {
        opened_files[i]->setSelected(opened_files[i]->view->isSelected());
    }
    ui->listWidget->blockSignals(false);
    ui->actionSave->setEnabled(ui->listWidget->selectedItems().size() == 1 && ((GeoTIFFWrapper*)ui->listWidget->selectedItems()[0])->isEdited());
    ui->actionSave_2->setEnabled(ui->listWidget->selectedItems().size() == 1 && ((GeoTIFFWrapper*)ui->listWidget->selectedItems()[0])->isEdited());
    ui->actionClose_File->setEnabled(ui->listWidget->selectedItems().size() == 1);
}

void MainWindow::on_listWidget_itemSelectionChanged()
{
    viewer->scene->blockSignals(true);
    for(int i = 0; i < opened_files.size(); ++i)
    {
        opened_files[i]->view->setSelected(opened_files[i]->isSelected());
    }
    viewer->scene->blockSignals(false);
    ui->actionSave->setEnabled(ui->listWidget->selectedItems().size() == 1 && ((GeoTIFFWrapper*)ui->listWidget->selectedItems()[0])->isEdited());
    ui->actionSave_2->setEnabled(ui->listWidget->selectedItems().size() == 1 && ((GeoTIFFWrapper*)ui->listWidget->selectedItems()[0])->isEdited());
    ui->actionClose_File->setEnabled(ui->listWidget->selectedItems().size() == 1);
}

void MainWindow::on_comboBox_2_currentTextChanged(const QString &arg1)
{
    if(arg1 == "Identity")
    {
        ui->spinBox_1->setValue(0);
        ui->spinBox_2->setValue(0);
        ui->spinBox_3->setValue(0);
        ui->spinBox_4->setValue(0);
        ui->spinBox_5->setValue(1);
        ui->spinBox_6->setValue(0);
        ui->spinBox_7->setValue(0);
        ui->spinBox_8->setValue(0);
        ui->spinBox_9->setValue(0);
    }
    else if(arg1 == "Edge Detection 1")
    {
        ui->spinBox_1->setValue(1);
        ui->spinBox_2->setValue(0);
        ui->spinBox_3->setValue(-1);
        ui->spinBox_4->setValue(0);
        ui->spinBox_5->setValue(0);
        ui->spinBox_6->setValue(0);
        ui->spinBox_7->setValue(-1);
        ui->spinBox_8->setValue(0);
        ui->spinBox_9->setValue(1);
    }
    else if(arg1 == "Edge Detection 2")
    {
        ui->spinBox_1->setValue(0);
        ui->spinBox_2->setValue(-1);
        ui->spinBox_3->setValue(0);
        ui->spinBox_4->setValue(-1);
        ui->spinBox_5->setValue(4);
        ui->spinBox_6->setValue(-1);
        ui->spinBox_7->setValue(0);
        ui->spinBox_8->setValue(-1);
        ui->spinBox_9->setValue(0);
    }
    else if(arg1 == "Edge Detection 3")
    {
        ui->spinBox_1->setValue(-1);
        ui->spinBox_2->setValue(-1);
        ui->spinBox_3->setValue(-1);
        ui->spinBox_4->setValue(-1);
        ui->spinBox_5->setValue(8);
        ui->spinBox_6->setValue(-1);
        ui->spinBox_7->setValue(-1);
        ui->spinBox_8->setValue(-1);
        ui->spinBox_9->setValue(-1);
    }
    else if(arg1 == "Sharpen")
    {
        ui->spinBox_1->setValue(0);
        ui->spinBox_2->setValue(-1);
        ui->spinBox_3->setValue(0);
        ui->spinBox_4->setValue(-1);
        ui->spinBox_5->setValue(5);
        ui->spinBox_6->setValue(-1);
        ui->spinBox_7->setValue(0);
        ui->spinBox_8->setValue(-1);
        ui->spinBox_9->setValue(0);
    }
    else if(arg1 == "Box Blur")
    {
        ui->spinBox_1->setValue(1);
        ui->spinBox_2->setValue(1);
        ui->spinBox_3->setValue(1);
        ui->spinBox_4->setValue(1);
        ui->spinBox_5->setValue(1);
        ui->spinBox_6->setValue(1);
        ui->spinBox_7->setValue(1);
        ui->spinBox_8->setValue(1);
        ui->spinBox_9->setValue(1);
    }
    else if(arg1 == "Gaussian Blur")
    {
        ui->spinBox_1->setValue(1);
        ui->spinBox_2->setValue(2);
        ui->spinBox_3->setValue(1);
        ui->spinBox_4->setValue(2);
        ui->spinBox_5->setValue(4);
        ui->spinBox_6->setValue(2);
        ui->spinBox_7->setValue(1);
        ui->spinBox_8->setValue(2);
        ui->spinBox_9->setValue(1);
    }
}

void MainWindow::on_InterSum_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 1);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterSub_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 2);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterMul_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 3);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterDiv_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 4);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterLess_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 5);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterLessEq_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 6);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterGt_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 7);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterGtEq_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 8);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterEq_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 9);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterNEq_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 10);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterAnd_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 11);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterOr_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 12);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterXor_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 13);
    thr.detach();

    load->exec();
}

void MainWindow::on_InterEquiv_clicked()
{
    std::thread thr(&MainWindow::interSecThr, this, 14);
    thr.detach();

    load->exec();
}

void MainWindow::on_actionClose_File_triggered()
{
    for (int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->isSelected())
        {
            GeoTIFFWrapper* tmp = opened_files[i];
            opened_files.erase(opened_files.begin() + i);
            delete tmp;
        }
    }
}

void MainWindow::on_actionSave_2_triggered()
{
    for (int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->isSelected())
            opened_files[i]->save();
    }
    ui->actionSave->setEnabled(false);
    ui->actionSave_2->setEnabled(false);
}



void MainWindow::on_actionSave_triggered()
{
    QString fn = QFileDialog::getSaveFileName(this, "Save File", "untitled.tif", "GeoTIFF File (*.tif *.tiff)");
    for (int i = 0; i < opened_files.size(); ++i)
    {
        if(opened_files[i]->isSelected())
            opened_files[i]->save(fn);
    }
    ui->actionSave->setEnabled(false);
    ui->actionSave_2->setEnabled(false);
}
