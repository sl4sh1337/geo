#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <vector>
#include <map>
#include "declares.h"
#include "multiviewer.h"
#include "geotiffwrapper.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    GeoTIFFWrapper* last_edited;
    uint32* buffered_tif_data = nullptr;
    int buffered_tif_data_width, buffered_tif_data_height;

    std::vector<GeoTIFFWrapper*> opened_files;
    std::map<QString, GeoTIFFWrapper*> name_to_obj;
    void update_tif(GeoTIFFWrapper* item, uint32* new_data, int new_width, int new_height);
    MultiViewer* viewer;
    QDialog* load;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void change_toggle();
    void kernelEachThr(int* kernel);
    void openFileThr(const QString& filename);
    void InterpolateThr(double s);
    void singularThr(int type);
    void interSecThr(int type);

private slots:
    void on_pushButton_clicked();

    void on_actionOpen_triggered();

    void on_Interpolation_clicked();

    void on_actionSave_triggered();

    void on_actionImage_Window_toggled(bool arg1);

    void on_MultiViewerFinished(int r);

    void closeLoad(bool);

    void on_abs_clicked();

    void on_round_clicked();

    void on_floor_clicked();

    void on_ceiling_clicked();

    void on_sqrt_clicked();

    void on_log_clicked();

    void on_exp_clicked();

    void on_cos_clicked();

    void on_sin_clicked();

    void on_orderChanged(const QModelIndex & sourceParent, int sourceStart, int sourceEnd, const QModelIndex & destinationParent, int destinationRow);

    void selectFiles();

    void on_listWidget_itemSelectionChanged();

    void on_InterSum_clicked();

    void on_comboBox_2_currentTextChanged(const QString &arg1);

    void on_actionUndo_triggered();

    void on_InterSub_clicked();

    void on_InterMul_clicked();

    void on_InterDiv_clicked();

    void on_InterLess_clicked();

    void on_InterLessEq_clicked();

    void on_InterGt_clicked();

    void on_InterGtEq_clicked();

    void on_InterEq_clicked();

    void on_InterNEq_clicked();

    void on_InterAnd_clicked();

    void on_InterOr_clicked();

    void on_InterXor_clicked();

    void on_InterEquiv_clicked();

    void on_actionClose_File_triggered();

    void on_actionSave_2_triggered();

signals:
    void loaded(bool);

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
