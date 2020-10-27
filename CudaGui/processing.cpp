#include "processing.h"
#include "ui_processing.h"

Processing::Processing(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Processing)
{
    ui->setupUi(this);
}

Processing::~Processing()
{
    delete ui;
}
