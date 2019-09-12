# -*- mode: ruby -*-
# vi: set ft=ruby :

$bootstrap = <<END
#!/bin/bash --login
ln -s /vagrant ~/geomesh
cd ~/geomesh
git lfs install
python -m venv .geomesh_env
source .geomesh_env/bin/activate
./setup.py install_deps
./setup.py develop
echo "source ~/geomesh/.geomesh_env/bin/activate" >> ~/.zshrc
echo "cd ~/geomesh" >> ~/.zshrc
END

Vagrant.configure("2") do |config|
  config.vm.box = "archlinux/archlinux"
  config.vm.provider "virtualbox" do |v|
    v.memory = 16384
    v.cpus = 8
  end
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true
  config.vm.provision :shell, privileged: true,
    inline: "pacman -Syu git git-lfs zsh grml-zsh-config base-devel cmake python python-setuptools gdal systemd-swap --noconfirm"
  config.vm.provision :shell, privileged: true,
    inline: "systemctl enable --now systemd-swap"
  config.vm.provision :shell, privileged: true,
    inline: "chsh -s /bin/zsh vagrant"
  config.vm.provision :shell, privileged: true,
    inline: "sed 's/#X11Forwarding no/X11Forwarding yes/g' /etc/ssh/sshd_config > /etc/ssh/sshd_config"
  config.vm.provision :shell, privileged: true, inline: "systemctl restart sshd"
  config.vm.provision :shell, privileged: false,
    inline: "echo \"" << $bootstrap <<"\" >> /tmp/bootstrap.sh"
  config.vm.provision :shell, privileged: false, inline: "/bin/bash --login /tmp/bootstrap.sh"
end
