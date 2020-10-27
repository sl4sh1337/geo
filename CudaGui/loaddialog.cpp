#include "loaddialog.h"
#include "ui_loaddialog.h"
#include <QMovie>

LoadDialog::LoadDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::LoadDialog)
{
    ui->setupUi(this);
    setWindowFlag(Qt::FramelessWindowHint);
    QMovie *movie = new QMovie(":/download.gif");
    movie->setScaledSize(QSize(ui->label->width(), ui->label->height()));
    ui->label->setMovie(movie);
    movie->start();
}

LoadDialog::~LoadDialog()
{
    delete ui;
}
